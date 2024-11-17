import functools
from typing import Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModel, TrainingArguments, DataCollatorForSeq2Seq, EvalPrediction, \
    AutoModelForCausalLM, Trainer, GenerationConfig, Seq2SeqTrainingArguments
import torch
from datasets import *
import numpy as np
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel, get_peft_config
from rouge_chinese import Rouge
from typing import Annotated, Any, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
import deepspeed
# accelerate launch train.py --config_file default_config.yaml

def process_once(example, tokenizer):
    '''
    none
    '''
    print(example)
    max_length = 1024
    input_ids = [151331, 151333]
    attention_mask = [1, 1]
    position_ids = list(range(len(input_ids)))
    loss_masks = [False, False]
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    images = []

    if example.get('image'):
        example['image'] = Image.open(example['image']).convert('RGB')
    else:
        example['image'] = img
    # =================================
    # user First treatment
    loss_mask_val = False
    user_model_new_temp = {"role": "user", "content": "".join([example['instruction'], example['input']]).strip(),
                           "image": example['image']}
    new_input_ids_all = tokenizer.apply_chat_template(
        [user_model_new_temp],
        tokenize=True,
        return_dict=True,
        padding=True
    )
    new_input_ids = new_input_ids_all['input_ids'][0][2:]
    new_attention_mask = new_input_ids_all['attention_mask'][0][2:]
    new_position_ids = list(range(position_ids[-1] + 1, position_ids[-1] + 1 + len(new_input_ids)))
    images.append(new_input_ids_all['images'])
    new_loss_masks = [loss_mask_val] * len(new_input_ids)
    input_ids += new_input_ids
    attention_mask += new_attention_mask
    position_ids += new_position_ids
    loss_masks += new_loss_masks
    del loss_mask_val, new_loss_masks, new_attention_mask, new_position_ids, user_model_new_temp, new_input_ids, new_input_ids_all
    # =================================
    # assistant First treatment
    loss_mask_val = True
    assistant_model_new_temp = {"role": "assistant", "content": example['output']}
    new_input_ids_all = tokenizer.apply_chat_template(
        [assistant_model_new_temp],
        tokenize=True,
        return_dict=True,
        padding=True
    )
    new_input_ids = new_input_ids_all['input_ids'][0][2:]
    new_attention_mask = new_input_ids_all['attention_mask'][0][2:]
    new_position_ids = list(range(position_ids[-1] + 1, position_ids[-1] + 1 + len(new_input_ids)))
    new_loss_masks = [loss_mask_val] * len(new_input_ids)
    input_ids += new_input_ids
    attention_mask += new_attention_mask
    position_ids += new_position_ids
    loss_masks += new_loss_masks
    # =================================
    input_ids.append(151336)  # EOS
    attention_mask.append(1)
    position_ids.append(len(position_ids))
    loss_masks.append(False)
    labels = []
    for input_id, mask in zip(input_ids, loss_masks):
        if mask:
            labels.append(input_id)
        else:
            labels.append(-100)
    return {
        'input_ids': input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "position_ids": position_ids[:max_length],
        "images": images[0][0],
        'labels': labels[:max_length]
    }


def process_run(examples, tokenizer):
    input_ids = []
    attention_mask = []
    position_ids = []
    labels = []
    images = []
    print(examples)
    # exit()

    for example_len in range(0, len(examples['input']) - 1):
        # print(example)
        date = process_once(
            {"input": examples["input"][example_len], "instruction": examples["instruction"][example_len],
             "output": examples["output"][example_len]}, tokenizer)
        input_ids.append(date['input_ids'])
        attention_mask.append(date['attention_mask'])
        position_ids.append(date['position_ids'])
        labels.append(date['labels'])
        images.append(date['images'])
    return {
        'input_ids': input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "images": images,
        'labels': labels,
    }


def compute_metrics(eval_preds: EvalPrediction, tokenizer):
    batched_pred_ids, batched_label_ids = eval_preds
    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        metrics_dct['bleu-4'].append(
            sentence_bleu([label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method3))
    return {k: np.mean(v) for k, v in metrics_dct.items()}
    # return [process_once(example,tokenizer) for example in examples]


class Seq2SeqTrainer(_Seq2SeqTrainer):
    # Not Support for apex
    def training_step(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        detached_loss = loss.detach() / self.args.gradient_accumulation_steps
        del inputs
        torch.cuda.empty_cache()
        return detached_loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict,
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        with torch.no_grad():
            if self.args.predict_with_generate:
                output_ids = inputs.pop('output_ids', None)
            loss, generated_tokens, labels = super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                **gen_kwargs
            )

            if generated_tokens is not None:
                generated_tokens = generated_tokens[:, inputs["input_ids"].size()[1]:]

            if self.args.predict_with_generate:
                labels = output_ids

            del inputs, output_ids
            torch.cuda.empty_cache()

        return loss, generated_tokens, labels


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = ([feature['output_ids'] for feature in features] if 'output_ids' in features[0].keys() else None)
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        return super().__call__(features, return_tensors)


if __name__ == '__main__':
    ds = Dataset.from_json('test_a.json', )
    print(ds)
    tokenizer = AutoTokenizer.from_pretrained("/home/xongrong/ai_test/model/ZhipuAI/glm-4v-9b-3", trust_remote_code=True, )
    img = Image.new('L', (224, 224), 0).convert('RGB')
    img.save("test.jpg")
    tokenized_ds = ds.map(
        functools.partial(
            process_run,
            tokenizer=tokenizer
        ),
        batched=True,
        remove_columns=ds.column_names,
        num_proc=1,
        writer_batch_size=1000,
        batch_size=1000,


    )

    print(tokenizer.decode(tokenized_ds[0]['input_ids']))
    print("----------")
    print(tokenized_ds[0]['input_ids'])
    print('----------')
    print(tokenized_ds[0]['labels'])
    print('----------')
    print(tokenizer.decode(tokenized_ds[0]['attention_mask']))
    # tokenizer=tokenizer.to("cuda:0")
    config2 = LoraConfig(target_modules=['query_key_value'], lora_dropout=0.1, lora_alpha=32, r=8,
                         task_type="CAUSAL_LM",
                         peft_type="LORA")
    # accelerate launch train.py --config_file
    print("sadfasafas")
    args = Seq2SeqTrainingArguments(
        num_train_epochs=20,
        output_dir="./output",
        learning_rate=5e-4,
        per_device_train_batch_size=1,
        dataloader_num_workers=16,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=500,
        log_level="info",
        logging_strategy="steps",
        logging_steps=1,
        per_device_eval_batch_size=4,
        eval_steps=500,

        deepspeed="ds_zero.txt",
        generation_config=GenerationConfig(max_new_tokens=512)
    )
    model2 = AutoModelForCausalLM.from_pretrained(
        "/home/xongrong/ai_test/model/ZhipuAI/glm-4v-9b-3",
        trust_remote_code=True,
        empty_init=False,
        use_cache=False,
        resume_download=True,
        torch_dtype=torch.bfloat16  # Must use BFloat 16

    )
    model = get_peft_model(model2, config2)
    model.print_trainable_parameters
    # ------------------------------------------------------------
    if True:
        for param in model.transformer.vision.parameters():  #
            param.requires_grad = False  #
    # ------------------------------------------------------------
    model.enable_input_require_grads()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    model.print_trainable_parameters()
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=tokenized_ds,
        eval_dataset=tokenized_ds,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )
    print(trainer.place_model_on_device)
    trainer.train()
    trainer.save_model(output_dir="./output_model")
