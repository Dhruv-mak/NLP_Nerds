import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch import nn
from box import Box
from pydantic import BaseModel
import json
from model import Record, paraDetox, Model

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
        gpt_ppl.append(100 if torch.isnan(loss_tensor) else torch.exp(loss_tensor))
    return gpt_ppl


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

def paradetox(sentence):
    tokenizer = AutoTokenizer.from_pretrained("HamdanXI/bart-base-paradetox-split")
    model = AutoModelForSeq2SeqLM.from_pretrained("HamdanXI/bart-base-paradetox-split")
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model.generate(**inputs)
    paradetoxed_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paradetoxed_sentence

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
    
def next_token(sentence, k, model_name):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

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


def select_nexttoken(sentence, k, model_name, mil_model):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    mil_model = load_mil_model('checkpoint_epoch_1.pth')
    language_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    potential_next_tokens = next_token(sentence, k, model_name)
    toxic_scores = []
    for token, prob in potential_next_tokens:
        new_sentence = sentence + " " + token
        toxic_scores.append([token, predict_toxicity(new_sentence, mil_model, tokenizer, language_model)])
    # change the toxicity score to 0 if toxicity score is less than 0.1
    threshold = 0.1
    for i in range(len(toxic_scores)):
        if toxic_scores[i][1] < threshold:
            toxic_scores[i][1] = 0
    # get the sum of the toxicity score
    sum_toxicity = sum([x[1] for x in toxic_scores])
    # divide every toxicity score by the sum of the toxicity score
    for i in range(len(toxic_scores)):
        toxic_scores[i][1] = toxic_scores[i][1] / sum_toxicity
    
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
    # Encode the incomplete sentence
    input_ids = tokenizer.encode(incomplete_sentence, return_tensors="pt")

    # Generate the rest of the sentence until the end-of-sentence token is encountered
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens  # Specify the maximum number of new tokens to generate
        )

    # Decode the generated output and the original input
    generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    original_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Remove the original input from the generated output to get only the new part
    new_part = generated_output[len(original_input):].strip()
    return new_part

    
def generate_detoxified_sentence(sentence, k, model_name, tokenizer, language_model):
    end_of_sentence_tokens = [".", "?", "!"]
    original_length = len(sentence.split())  # Get the number of words in the original sentence
    while True:
        next_token = select_nexttoken(sentence, k, model_name, "checkpoint_epoch_1.pth")
        sentence += " " + next_token
        if next_token in end_of_sentence_tokens:
            break
    generated_part = ' '.join(sentence.split()[original_length:])  # Get only the generated part
    return generated_part


def get_diversity_score(sentence, tokenizer, model):
    # Tokenize the sentence
    input_ids = tokenizer.encode(sentence, return_tensors="pt")

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

if __name__ == '__main__':
    file_path = 'realtoxicityprompts-data/prompts.jsonl'
    prompts, labels = load_data(file_path)
    mil_model = load_mil_model('checkpoint_epoch_1.pth')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    language_model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
    for i in range(5):
        r = Record(pre_prompt=prompts[i])
        r.toxicity = predict_toxicity(prompts[i], mil_model, tokenizer, language_model)
        # r.calc_flair_ppl = calc_flair_ppl(prompts[i])
        r.calc_gpt_ppl = calc_gpt_ppl(prompts[i])
        # r.do_cola_eval = do_cola_eval(prompts[i])
        r.diversity = get_diversity_score(prompts[i], tokenizer, language_model)
        para_obj = paraDetox()
        para_obj.com_prompt = complete_sentence(prompts[i], tokenizer, language_model)
        para_obj.paraphrase = paradetox(para_obj.com_prompt)
        para_obj.com_toxicity = predict_toxicity(prompts[i]+para_obj.com_prompt, mil_model, tokenizer, language_model)
        para_obj.para_toxicity = predict_toxicity(para_obj.paraphrase, mil_model, tokenizer, language_model)
        para_obj.calc_gpt_ppl = calc_gpt_ppl(para_obj.paraphrase)
        para_obj.diversity = get_diversity_score(para_obj.paraphrase, tokenizer, language_model)
        r.paraphrase = para_obj
        r.models_out = []
        for model_name in ["gpt2", "gpt2-medium", "gpt2-large"]:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            gen = generate_detoxified_sentence(prompts[i], 5, model_name, tokenizer, language_model)
            model_obj = Model(model_name=model_name, com_prompt=gen)
            model_obj.overall_toxicity = predict_toxicity(gen, mil_model, tokenizer, language_model)
            model_obj.com_toxicity = predict_toxicity(prompts[i]+gen, mil_model, tokenizer, language_model)
            # model_obj.calc_flair_ppl = calc_flair_ppl(new_sentence)
            model_obj.calc_gpt_ppl = calc_gpt_ppl(gen)
            # model_obj.do_cola_eval = do_cola_eval(new_sentence)
            # model_obj.gm = get_gm(args, accuracy, emb_sim, char_ppl)
            # model_obj.joint = get_j(args, accuracy_by_sent, similarity_by_sent, cola_stats, preds)
            r.models_out.append(model_obj)
        json_obj = r.model_dump()
        # write json object to file in jsonl
        with open('output.jsonl', 'a') as f:
            json.dump(json_obj, f)
            f.write('\n')
