from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from tqdm import tqdm

translator_paths = {'es': 'Helsinki-NLP/opus-mt-en-es', 'hi': 'Helsinki-NLP/opus-mt-en-hi'}
translators = {lg: {"tokenizer": MarianTokenizer.from_pretrained(path), 
                    "model": MarianMTModel.from_pretrained(path).cuda()}
               for lg, path in translator_paths.items()}

print('Translators loaded.')


def translate(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model.generate(inputs.input_ids.cuda(), num_beams=5, early_stopping=True,num_return_sequences=1)
    return [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False) for output in outputs]


data = load_dataset("tweet_eval", "sentiment")

def translate_add_columns(example, translators):
    for lg in translators.keys():
        tokenizer = translators[lg]['tokenizer']
        model = translators[lg]['model']
        example[lg] = translate(example['text'], tokenizer, model)[0]
    return example

print(data['test'].select([0]).map(translate_add_columns, fn_kwargs={'translators': translators})[0])

data['test'].map(translate_add_columns, fn_kwargs={'translators': translators}).save_to_disk('data/tweeteval_sentiment_translated_test')

data['validation'].map(translate_add_columns, fn_kwargs={'translators': translators}).save_to_disk('data/tweeteval_sentiment_translated_validation')
