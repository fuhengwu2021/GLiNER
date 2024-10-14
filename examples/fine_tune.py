import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import torch
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator
import hiq

train_path = "data.json"

with open(train_path, "r") as f:
    data = json.load(f)
print('Dataset size:', len(data))
for i in data:
    if len(i.keys()) > 2:
        print(i.keys())

# random.shuffle(data)
# print('Dataset is shuffled...')

train_dataset = data[:int(len(data) * 0.9)]
test_dataset = data[int(len(data) * 0.9):]

print('Dataset is split...')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = GLiNER.from_pretrained("urchade/gliner_small")

# use it for better performance, it mimics original implementation but it's less memory efficient
data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

# Optional: compile model for faster training
model = model.to(device)
hiq.print_model(model)

# calculate number of epochs
num_steps = 500
batch_size = 8
data_size = len(train_dataset)
num_batches = data_size // batch_size
num_epochs = max(1, num_steps // num_batches)

training_args = TrainingArguments(
    output_dir="models",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="linear",  # cosine
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    focal_loss_alpha=0.75,
    focal_loss_gamma=2,
    num_train_epochs=num_epochs,
    evaluation_strategy="steps",
    save_steps=100,
    save_total_limit=10,
    dataloader_num_workers=0,
    use_cpu=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=model.data_processor.transformer_tokenizer,
    data_collator=data_collator,
)

trainer.train()
# trained_model = GLiNER.from_pretrained("models/checkpoint-100", load_tokenizer=True)
