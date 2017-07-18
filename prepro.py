from scipy import ndimage
from collections import Counter
from core.utils import save_pickle
import numpy as np
import pandas as pd
import hickle
import os
import json


def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]}
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        if image_id in id_to_filename:
            pp = os.path.join(image_dir, id_to_filename[image_id])
            cap = _pro_caption(annotation["caption"])
            if len(cap.split(" ")) <= max_length+2:
                data.append({"image_path":pp, "caption":cap})
    return data
def _pro_caption(caption):
    caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
    caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
    caption = caption.lower()
    caption = " ".join(['<START>']+caption.split()+['<END>'])  # replace multiple spaces
    return caption


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations):
        caption = caption['caption']
        words = caption.split(' ')  # caption contrains only lower-case words
        for w in words:
            if w != '<START>' and w !='<END>':
                counter[w] += 1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<EOP>': 3}
    idx = 4
    for word in vocab:
        print word
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx

def save_json(data, path):
    with open(path, 'wb') as f:
        json.dump(data, f)
        print ('Saved %s..' % path)

def main():
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.
    max_length = 15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 5

    # about 40000 images and 200000 captions
    val_dataset = _process_caption_data(
        caption_file='image/annotations/captions_val2014.json',
        image_dir='val2014',
        max_length=max_length)

    # about 80000 images and 400000 captions for train dataset
    train_dataset = _process_caption_data(
        caption_file='image/annotations/captions_train2014.json',
        image_dir='train2014',
        max_length=max_length)

    # about 4000 images and 20000 captions for val / test dataset
    val_cutoff = int(0.1 * len(val_dataset))
    test_cutoff = int(0.1 * len(val_dataset))

    print 'Finished processing caption data'

    dataset = {}
    dataset['train'] = train_dataset+val_dataset[:len(val_dataset)-test_cutoff-val_cutoff]
    dataset['test'] = val_dataset[len(val_dataset)-test_cutoff:]
    dataset['val'] = val_dataset[len(val_dataset)-test_cutoff-val_cutoff:len(val_dataset)-test_cutoff]

    for split in ['train', 'val', 'test']:
        save_json(dataset[split], "data/%s.json"%split)
        if split == 'train':
            word_to_idx = _build_vocab(annotations=dataset[split], threshold=word_count_threshold)
            save_pickle(word_to_idx, 'data/word_to_idx.pkl')
if __name__ == "__main__":
    main()