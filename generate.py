
import torch, sys
import numpy as np
from model import LEDKForConditionalGeneration, GraphEncoder
from transformers import AutoTokenizer
from utils import load_dataset, load_train_config, get_processed_elife_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Config
model_dir = sys.argv[1]

config = load_train_config(model_dir)
config['batch_size'] = 16

# Data
ds = "elife"
test = load_dataset(ds, "test")
test = [{"idx": i, **x} for i, x in enumerate(test)]

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
test_dataloader = get_processed_elife_data(ds, tokenizer, config, "test", shuffle=False)

# load model 
model = LEDKForConditionalGeneration.from_pretrained(
    model_dir, 
    torch_dtype=torch.float16, 
    is_merge_encoders=config['is_merge_encoders'], 
    is_graph_decoder=config['is_graph_decoder']
    ).to(device)
graph_encoder = GraphEncoder(config)

# Eval loop
model.eval()
preds = []
for step, batch in enumerate(test_dataloader):
    with torch.no_grad():
        print(step)
        aids = batch["idx"]
            
        graph_enc_out = []
        for i, aid in enumerate(aids):
            graph_out = graph_encoder.forward(aid, "test", device)
            graph_out = torch.nn.functional.pad(graph_out, (0,0,0,1024-graph_out.shape[0]), "constant", 0)
            graph_enc_out.append(graph_out)
        graph_enc_out = torch.stack(graph_enc_out, dim=1).to(torch.float16)
        graph_enc_out = graph_enc_out.view(len(aids), -1, 1024)
            
        del batch['idx']
           
        generated_tokens = model.generate(
            batch["input_ids"].to(device),
            graph_encoder_outputs=graph_enc_out.to(device), 
        )

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]

        generated_tokens = generated_tokens.cpu().numpy()
        decoded_preds = np.where(generated_tokens != -100, generated_tokens, tokenizer.pad_token_id)
            
        decoded_preds = tokenizer.batch_decode(decoded_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        preds.extend(decoded_preds)

with open(model_dir+"/preds.txt", "w") as out_f:
    for p in preds:
        out_f.write(p+"\n")

            

