import torch
import pandas as pd
import config

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm


mapper = {
    '0': 0,
    '1': 0.25,
    '2': 0.5,
    '3': 0.75,
    '4': 1
}


class P2PDataset(Dataset):
    def __init__(self, tokenizer, data):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        
        self.inputs = []
        
        self._build()
        
    def __len__(self):
        return len(self.inputs)
        
    def _build_context(self, p1, p2, context):
        prompt = f'context: {context}. Are {p1}: {p2} related?'
        return prompt
        
    def _build(self):
        for phrase_1, phrase_2, context, in zip(
            self.data['anchor'], self.data['target'], self.data['description']):
            context = self._build_context(phrase_1, phrase_2, context)
            input_token_ids = self.tokenizer.encode_plus(
                context, padding='max_length', max_length=512, return_tensors='pt')
            self.inputs.append(input_token_ids)
            
    def __getitem__(self, index):
        input_ids = self.inputs[index]['input_ids'].squeeze()
        attention_mask = self.inputs[index]['attention_mask'].squeeze()
        token_type_ids = self.inputs[index]['token_type_ids'].squeeze()
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'token_type_ids': token_type_ids
            }


def infer():
    model_path = config.MODEL_PATH['bert_for_patents']
    input_path = config.INPUT_FOLDER_PATH

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print('Device : ', device)

    test_data = pd.read_csv(f'{input_path}/test.csv')
    cpc_scheme_data = pd.read_csv(f'{input_path}/cpc_scheme_data.csv')
    cpc_scheme_data = cpc_scheme_data.rename({'classification': 'context', 'description': 'description'}, axis=1)
    test_data = test_data.merge(cpc_scheme_data, how='left', on=['context'])

    test_dataset = P2PDataset(tokenizer, test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=2)

    outputs = []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            preds = model.forward(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device))
            
        logits = preds.logits
        prediction_class_ids = [logit.argmax().item() for logit in logits]
        prediction_class = [mapper[str(_id)] for _id in prediction_class_ids]
        print(prediction_class)
        outputs.extend(prediction_class)
    return outputs


if __name__ == "__main__":
    y_preds = infer()
    sample_submission = pd.read_csv('/Users/Nester/Documents/Career Management/github/US Patent Phrase to Phrase Matching/kaggle-usp2pmatching/input/sample_submission.csv')
    print(sample_submission)
    sample_submission.score = y_preds