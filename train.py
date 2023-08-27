from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import Adafactor
from datasets import load_dataset
import torch

from datetime import datetime
import pandas as pd
import random

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", legacy=False)
tokenizer.add_tokens(['<']) # TODO Remove a space from after every '<'

base_dir = "data/"
# If colab
# base_dir = '/content/'
data_path = base_dir + 'time_seed_fhir.csv'

fhir_dataset = load_dataset('csv', data_files=data_path, split='train')
fhir_dataset = fhir_dataset.train_test_split(test_size=0.2)
print(f"Training Samples: {fhir_dataset['train'].num_rows}, Test Samples: {fhir_dataset['test'].num_rows}")

prefix = "Generate a FHIR data row using the seed: "
# FHIR,seed
max_source_length = 64
max_target_length = 4096

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["seed"]]

    model_inputs = tokenizer(
        text_target      = inputs,
        padding          = "longest",
        max_length       = max_source_length,
        truncation       = True,
    )
    # Setup the tokenizer for targets
    labels = tokenizer(
        text_target      = examples["FHIR"],
        padding          = "longest",
        max_length       = max_target_length,
        truncation       = True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = fhir_dataset.map(preprocess_function, batched=True).remove_columns(['seed', 'FHIR'])

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")
model.resize_token_embeddings(len(tokenizer))

# input_text = "[resourceType] CarePlan [id] 76d3b4b2-398f-4373-4957-599959c26e87 [meta][profile][0] http://hl7.org/fhir/us/core/StructureDefinition/us-core-careplan [text][status] generated [text][div] <div xmlns=""http://www.w3.org/1999/xhtml"">Care Plan for Infectious disease care plan (record artifact).<br/>Activities: <ul><li>Infectious disease care plan (record artifact)</li><li>Infectious disease care plan (record artifact)</li></ul><br/>Care plan is meant to treat COVID-19.</div> [status] completed [intent] order [category][0][coding][0][system] http://hl7.org/fhir/us/core/CodeSystem/careplan-category [category][0][coding][0][code] assess-plan [category][1][coding][0][system] http://snomed.info/sct [category][1][coding][0][code] 736376001 [category][1][coding][0][display] Infectious disease care plan (record artifact) [category][1][text] Infectious disease care plan (record artifact) [subject][reference] Patient/4a62b5fe-0dbd-fef9-e9a9-c21cecc5df61 [encounter][reference] Encounter/3bffc09e-9247-6538-e8db-6c119e3f1e2c [period][start] 2020-03-09T16:39:52-04:00 [period][end] 2020-03-27T13:39:52-04:00 [careTeam][0][reference] CareTeam/51954d07-f4a7-0f1a-b7a3-9c0f1f592038 [addresses][0][reference] Condition/3e78ad09-6fec-ede1-b27a-47e79bee28f1 [activity][0][detail][code][coding][0][system] http://snomed.info/sct [activity][0][detail][code][coding][0][code] 409524006 [activity][0][detail][code][coding][0][display] Airborne precautions (procedure) [activity][0][detail][code][text] Airborne precautions (procedure) [activity][0][detail][status] completed [activity][0][detail][location][display] BERKSHIRE MEDICAL CENTER INC - 1 [activity][1][detail][code][coding][0][system] http://snomed.info/sct [activity][1][detail][code][coding][0][code] 361235007 [activity][1][detail][code][coding][0][display] Isolation of infected patient (procedure) [activity][1][detail][code][text] Isolation of infected patient (procedure) [activity][1][detail][status] completed [activity][1][detail][location][display] BERKSHIRE MEDICAL CENTER INC - 1"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))

#optimizer = Adafactor(model.parameters(), lr=0.001, scale_parameter=False, relative_step=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

training_args = Seq2SeqTrainingArguments(
    output_dir                  = "./results",
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    learning_rate               = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    weight_decay                = 0.01,
    save_total_limit            = 5,
    num_train_epochs            = 5,
)

trainer = Seq2SeqTrainer(
    model         = model,
    args          = training_args,
    train_dataset = tokenized_dataset['train'],
    eval_dataset  = tokenized_dataset['test'],
    tokenizer     = tokenizer,
    data_collator = data_collator,
    optimizers    = [optimizer, scheduler],
)

trainer.train()

seeds = []
n_outputs = 100

for x in range(n_outputs):
    seeds.append(datetime.now().strftime("%Y:%m:%d %H:%M:%S:%f") + " " + str(random.randint(0, 100_000_000_000_00)))

prefix = "Generate a FHIR data row using the seed: "
seeds = {
    "seed": [prefix + doc for doc in seeds]
}

model_outputs = trainer.model.generate(**
    tokenizer(
        text_target      = seeds['seed'],
        padding          = "longest",
        max_length       = max_source_length,
        truncation       = True,
        return_tensors    = 'pt'
    ).to('cuda'),
    max_length=4096
)


df = pd.DataFrame({"Generated_FHIR": tokenizer.batch_decode(model_outputs, skip_special_tokens=True)})
df.head(5)

df.to_csv("generated_fhir.csv", index=False)