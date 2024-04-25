import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch import nn
from pydantic import BaseModel
from typing import List
import json


class Model(BaseModel):
    models_name: str = ""
    com_prompt: str = ""
    gen_toxicity: float = 0
    overall_toxicity: float = 0
    com_toxicity: float = 0
    calc_gpt_ppl: float = 0
    diversity: float = 0
class paraDetox(BaseModel):
    com_prompt: str = ""
    paraphrase: str = ""
    com_toxicity: float = 0
    para_toxicity: float = 0
    calc_gpt_ppl: float = 0
    diversity: float = 0
class Record(BaseModel):
    pre_prompt: str
    toxicity: float = 0
    models_out: List[Model] = []
    calc_gpt_ppl: float = 0
    paraphrase: paraDetox = None
    diversity: float = 0
    
def load_data(file_path):
    prompts = []
    labels = []
    i = 0
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            prompt_text = data['prompt']['text']
            toxicity = data['prompt']['toxicity']
            if toxicity is not None:
                prompts.append(prompt_text)
                labels.append([toxicity])
            else:
                i += 1
    print("none vals:", i)
    return prompts, torch.tensor(labels)




class MILNetwork(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(MILNetwork, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h = self.gru(x)
        h = h.squeeze(0)
        out = self.fc(h)
        return self.sigmoid(out)

    

def next_token(sentence, k, model, tokenizer):
    # Encode the sentence
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    # Get the logits for the next token
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
    # Get the top k tokens and their probabilities
    top_probs, top_indices = torch.topk(torch.softmax(logits, dim=-1), k)
    # Convert token indices to tokens
    top_tokens = [tokenizer.decode([idx], skip_special_tokens=True).strip() for idx in top_indices.squeeze().tolist()]
    # Convert probabilities to a list
    top_probs = top_probs.squeeze().tolist()
    # Return the top k tokens with their probabilities
    return list(zip(top_tokens, top_probs))



def load_mil_model(model_path, hidden_size=768, output_size=1):
    mil_model = MILNetwork(hidden_size, output_size)
    checkpoint = torch.load(model_path)
    mil_model.load_state_dict(checkpoint['model_state_dict'])  # Load only the model_state_dict
    mil_model.eval()  # Set the model to evaluation mode
    return mil_model



def predict_toxicity(sentence, mil_model, tokenizer, language_model, max_length=50):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    language_model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
    # Set the pad_token for the tokenizer if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Tokenize and encode the sentence
    inputs = tokenizer(sentence, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # Get embeddings from the language model
    with torch.no_grad():
        outputs = language_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]  # Use hidden_states instead of last_hidden_state
    # Get the toxicity score from the MIL model
    with torch.no_grad():
        toxicity_score = mil_model(embeddings).squeeze().item()
    return toxicity_score



def select_nexttoken(sentence, k, mil_model, model, tokenizer):
    potential_next_tokens = next_token(sentence, k, model, tokenizer)
    toxic_scores = []
    for token, prob in potential_next_tokens:
        new_sentence = sentence + " " + token
        toxic_scores.append([token, predict_toxicity(new_sentence, mil_model, tokenizer, model)])
    # change the toxicity score to 0 if toxicity score is less than 0.1
    threshold = 0.1
    for i in range(len(toxic_scores)):
        if toxic_scores[i][1] < threshold:
            toxic_scores[i][1] = 0
    
    '''Change'''
    # get the sum of the toxicity score
    sum_toxicity = sum([x[1] for x in toxic_scores])

    # Check if sum_toxicity is zero
    if sum_toxicity > 0:
        # Normalize the toxicity scores
        for i in range(len(toxic_scores)):
            toxic_scores[i][1] = toxic_scores[i][1] / sum_toxicity
    else:
        # If sum_toxicity is zero, set all normalized scores to zero
        for i in range(len(toxic_scores)):
            toxic_scores[i][1] = 0
    '''Change End'''
    
    sum_prob = sum(x[1] for x in potential_next_tokens)
    # divide every probability by the sum of the probability
    for i in range(len(potential_next_tokens)):
        potential_next_tokens[i] = (potential_next_tokens[i][0], potential_next_tokens[i][1] / sum_prob)
    # now convert this P (y|x) = sof tmax(PLM (y|x) − λPtox.(y|x)) to code
    lambda_val = 0.5
    final_scores = []
    for i in range(len(potential_next_tokens)):
        final_scores.append((potential_next_tokens[i][0], potential_next_tokens[i][1] - lambda_val * toxic_scores[i][1]))
    # get the token with the highest final score
    max_score = -1
    max_token = ""
    for token, score in final_scores:
        if score > max_score:
            max_score = score
            max_token = token
    return max_token



def complete_sentence(incomplete_sentence, tokenizer, model, max_new_tokens=30):
    # Define end-of-sentence tokens
    end_of_sentence_tokens = ['.', '?', '!']
    # Encode the incomplete sentence
    input_ids = tokenizer.encode(incomplete_sentence, return_tensors="pt")
    # Generate the rest of the sentence until an end-of-sentence token is encountered
    generated_output = incomplete_sentence
    while True:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=1  # Generate one token at a time
            )
        # Decode the generated token
        generated_token = tokenizer.decode(output_ids[0][-1], skip_special_tokens=True)
        # Append the generated token to the output
        generated_output += ' ' + generated_token
        # Check if the generated token is an end-of-sentence token
        if generated_token in end_of_sentence_tokens:
            break
        # Update the input_ids for the next iteration
        input_ids = output_ids
    # Remove the original input from the generated output to get only the new part
    new_part = ' '.join(generated_output.split()[len(incomplete_sentence.split()):])
    return new_part.strip()



def generate_detoxified_sentence(sentence, k, mil_model, model, tokenizer):
    end_of_sentence_tokens = [".", "?", "!"]
    original_length = len(sentence.split())
    while True:
        next_token = select_nexttoken(sentence, k, mil_model, model, tokenizer)
        sentence += " " + next_token
        if next_token in end_of_sentence_tokens:
            break
    generated_part = ' '.join(sentence.split()[original_length:])
    return generated_part



def paradetox(sentence):
    tokenizer = AutoTokenizer.from_pretrained("HamdanXI/bart-base-paradetox-split")
    model = AutoModelForSeq2SeqLM.from_pretrained("HamdanXI/bart-base-paradetox-split")
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model.generate(**inputs)
    paradetoxed_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paradetoxed_sentence



def get_diversity_score(sentence, tokenizer, model):
    # Tokenize the sentence
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    '''Change'''
    if input_ids.nelement() == 0:
        return 0.0  # Return a default diversity score for empty input
    '''Change end'''
    # Get the hidden states of the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    # Calculate the diversity score
    diversity_score = 0
    for i in range(len(hidden_states) - 1):
        for j in range(i + 1, len(hidden_states)):
            diversity_score += torch.cosine_similarity(hidden_states[i], hidden_states[j], dim=-1).mean().item()
    return diversity_score



def calc_gpt_ppl(sentence):
    detokenize = lambda x: x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace("( ", "(")
    gpt_ppl = []
    gpt_model = AutoModelForCausalLM.from_pretrained('gpt2-medium')
    gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    gpt_model.eval()
    with torch.no_grad():
        sent = detokenize(sentence)
        if len(sent) == 1:
            sent = sent + '.'
        input_ids = gpt_tokenizer.encode(sent)
        inp = torch.tensor(input_ids).unsqueeze(0)
        try:
            result = gpt_model(inp, labels=inp, return_dict=True)
            loss = result.loss.item()
        except Exception as e:
            print(f'Got exception "{e}" when calculating gpt perplexity for sentence "{sent}" ({input_ids})')
            loss = 100
        loss_tensor = torch.tensor(loss)
        gpt_ppl = 100 if torch.isnan(loss_tensor) else torch.exp(loss_tensor).item() 
    return gpt_ppl