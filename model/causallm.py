from fastchat.conversation import get_conv_template
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import huggingface_hub
import torch
# huggingface_hub.login("")


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words:list, tokenizer, device):
        self.keywords = [torch.LongTensor(tokenizer.encode(w, add_special_tokens=False)[-5:]).to(device) for w in stop_words]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for k in self.keywords:
            if len(input_ids[0]) > len(k) and torch.equal(input_ids[0][-len(k):], k):
                return True
        return False


class ChatCausalLM:
    def __init__(
        self,
        model_name,
        max_new_tokens=512,
        temperature=0.7,
        device="auto",
        cache_dir=None,
        conversation_template=None,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = "cuda" if device=="auto" else device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            cache_dir=cache_dir
        )
        self.conversation_template = conversation_template

    def generate(self, messages, stop=None, print_prompt=False):
        human_role_set = {"user", "human"}
        ai_role_set = {"bot", "ai", "gpt", "assistant"}
        conv = get_conv_template(self.conversation_template)
        for message in messages:
            if message['role'] == 'system':
                conv.set_system_message(message['content'])
            else:
                conv.append_message(
                    conv.roles[0] if message['role'] in human_role_set else conv.roles[1],
                    message["content"]
                )
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if print_prompt:
            print(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        stop_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop, self.tokenizer, self.device)]) if stop else None
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            stopping_criteria=stop_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        inputs_token_length = len(inputs.input_ids[0])
        new_tokens = outputs[0][inputs_token_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if stop:
            for ending in stop:
                if text.endswith(ending):
                    text = text[:-len(ending)]
                    break

        return text.strip()


if __name__ == "__main__":
    model = ChatCausalLM(
        "meta-llama/Llama-2-7b-chat-hf",
        max_new_tokens=512,
        temperature=0.7,
        device="auto",
        conversation_template="llama-2"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    text = model.generate(messages, print_prompt=True)
    print(text)