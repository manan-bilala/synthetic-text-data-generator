%%writefile synthgen.py
import pandas as pd
import re
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Optional
from pydantic import BaseModel, create_model, ValidationError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv_data(df):
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    column_types = {}
    for col in df.columns:
        col_data = df[col].dropna()
        if col_data.empty:
            column_types[col] = str
        else:
            try:
                pd.to_numeric(col_data)
                if all(col_data.apply(lambda x: float(x).is_integer())):
                    column_types[col] = int
                else:
                    column_types[col] = float
            except:
                column_types[col] = str
    examples = []
    for _, row in df.dropna().sample(min(5, len(df))).iterrows():
        parts = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val): continue
            name = col.replace("_", " ").title()
            if column_types[col] == float:
                parts.append(f"{name}: ${float(val):.2f}")
            else:
                parts.append(f"{name}: {val}")
        examples.append("; ".join(parts))
    return examples, column_types

def create_dynamic_model(column_specs):
    fields = {col: (Optional[typ], None) for col, typ in column_specs.items()}
    return create_model("DynamicRecord", **fields)

def parse_generated_text(text, column_types):
    text = re.sub(r"\s+", " ", text.strip())
    matches = re.findall(r'([\w\s]+):\s*([^;]+)', text)
    if not matches:
        return None
    record = {}
    expected = list(column_types.keys())
    expected_titles = [c.replace('_', ' ').title() for c in expected]
    for key, val in matches:
        key_clean = key.strip().title()
        matched = next((c for c in expected_titles if key_clean.lower() in c.lower()), None)
        if matched:
            actual_key = matched.replace(" ", "_").lower()
            typ = column_types[actual_key]
            try:
                val = val.replace("$", "").replace(",", "").strip()
                if not val or val.lower() == 'nan':
                    return None
                if typ == int:
                    record[actual_key] = int(float(val))
                elif typ == float:
                    record[actual_key] = float(val)
                else:
                    record[actual_key] = str(val.strip())
            except:
                return None
    return record if record else None

def generate_synthetic_data(df, num_samples):
    examples, column_types = load_csv_data(df)
    DynamicModel = create_dynamic_model(column_types)
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    synthetic_data = []
    max_attempts = num_samples * 10
    attempts = 0
    creativity_phrases = [
        "Create a unique and diverse record:",
        "Invent a different, realistic patient entry:",
        "Generate a varied and creative record:",
        "Simulate a new and distinctive patient entry:"
    ]
    while len(synthetic_data) < num_samples and attempts < max_attempts:
        attempts += 1
        sampled_examples = random.sample(examples, min(3, len(examples)))
        example_text = "\n".join(sampled_examples)
        creativity_prompt = random.choice(creativity_phrases)
        field_list = ", ".join([col.replace('_', ' ').title() for col in column_types.keys()])
        prompt = (
            f"Generate realistic and diverse synthetic medical billing records with the following fields:\n"
            f"{field_list}\n\n"
            "Each record should follow this format:\n"
            "Field Name: Value; Field Name: Value; ...\n\n"
            "Examples:\n" + example_text +
            f"\n\n{creativity_prompt}\n"
        )
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)
        outputs = model.generate(input_ids, max_new_tokens=128, temperature=1.0, top_p=0.95, do_sample=True)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        parsed = parse_generated_text(output_text, column_types)
        if not parsed:
            logger.warning(f"❌ Record {len(synthetic_data)+1}: Failed to parse")
            continue
        try:
            validated = DynamicModel(**parsed)
            if any(val is None or str(val).strip() == "" for val in validated.model_dump().values()):
                continue
            synthetic_data.append(validated.model_dump())
            logger.info(f"✅ Record {len(synthetic_data)} added")
        except ValidationError:
            continue
    return pd.DataFrame(synthetic_data)