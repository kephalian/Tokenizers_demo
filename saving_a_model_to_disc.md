## SAVING A HUGGING FACE TRANSFORMER MODEL FROM HUGGING FACE CACHE TO DISC FOR FUTURE USE

from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", force_download=True)


tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]


inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")


outputs = model(**inputs)
print(outputs.last_hidden_state.shape)


## Note for the beginners like me, tokenizer is also like a model 

#### (not like a program, but like a file and has to be saved and loaded independently)


## pt_save_directory="/home/san/Models/distilbert-base-uncased-finetuned-sst-2-english"


## tokenizer.save_pretrained(pt_save_directory)


## model.save_pretrained(pt_save_directory)



