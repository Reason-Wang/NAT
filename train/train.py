import json
from dataclasses import field, dataclass
from typing import Optional
import huggingface_hub
import sys
sys.path.append('./')
from data.dataset import ConversationDataset, CollatorWithPadding
from data.utils import load_data, load_data_with_prompts, load_pos_neg_data_with_prompts
# with open('data/keys.json', 'r') as f:
#     keys = json.load(f)['huggingface_api_key']
# huggingface_hub.login("")
import torch
import transformers
from transformers import Trainer
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: str = field(default="google/flan-t5-base")
    model_max_length: int = 4096
    task_name: str = None
    template: str = None
    pos_path: str = field(default=None)
    neg_path: str = field(default=None)
    prompt_path: str = field(default=None)
    question_path: str = field(default=None)
    pos_num: int = field(default=1000)
    neg_num: int = field(default=1000)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size = 8
    learning_rate: float = 5e-5
    num_train_epochs: int = 3


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # data_list = load_data("data/AgentInstruct.json")
    # Use same data ratio as the original paper
    # data_list = load_agenttuning_data()[:args.num_data]
    # data_list = load_data(args.data_path)[:args.num_data]
    data_list = load_pos_neg_data_with_prompts(
        task_name=args.task_name,
        pos_path=args.pos_path,
        neg_path=args.neg_path,
        prompt_path=args.prompt_path,
        template=args.template,
        question_path=args.question_path,
        pos_num=args.pos_num,
        neg_num=args.neg_num,
    )

    print(f"Data length: {len(data_list)} Neg num: {args.neg_num}")

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        max_length=args.model_max_length,
        truncation=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = ConversationDataset(
        tokenizer=tokenizer,
        organized_data=data_list,
        conv_template_name="llama-2",
        task_name=args.task_name,
        max_length=args.model_max_length,
        positive_only=False,
    )

    collator = CollatorWithPadding(tokenizer=tokenizer)

    model = transformers.LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id

    trainer = Trainer(
        model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    train()