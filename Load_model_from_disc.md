# Loading a Hugging Face Transformer model and tokenizer from DISC
from transformers import AutoModel
from transformers import AutoTokenizer

model = AutoModel.from_pretrained("/home/san/Models/distilbert-base-uncased-finetuned-sst-2-english", force_download=False, local_files_only=True, low_cpu_mem_usage=True)
tokenizer=AutoTokenizer.from_pretrained("/home/san/Models/distilbert-base-uncased-finetuned-sst-2-english")
