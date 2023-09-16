import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from deidentify.corpus_loader import CORPUS_PATH, CorpusLoader
from deidentify.base import Corpus, Document
from deidentify.tokenizer import TokenizerFactory
from loguru import logger
from typing import List
from functools import partial
from os.path import join

import train_utils
import flair_utils

import flair.data
from flair.embeddings import (FlairEmbeddings, PooledFlairEmbeddings, StackedEmbeddings, TokenEmbeddings, WordEmbeddings)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

def _ignore_sentence(sent):
    return sent[0].text.startswith('===')

def _predict_ignored(sents):
    for sent in sents:
        for token in sent:
            token.add_tag('ner', 'O')

def make_predictions(tagger, filtered_corpus: flair_utils.FilteredCorpus):
    tagger.predict(filtered_corpus.train)
    tagger.predict(filtered_corpus.dev)
    tagger.predict(filtered_corpus.test)

    _predict_ignored(filtered_corpus.train_ignored)
    _predict_ignored(filtered_corpus.dev_ignored)
    _predict_ignored(filtered_corpus.test_ignored)

def get_embeddings(
    pooled: bool,
    contextual_forward_path: str = None,
    contextual_backward_path: str = None,
) -> List[TokenEmbeddings]:
    logger.info('Use English embeddings')
    word_embeddings = 'glove'
    contextual_forward = 'news-forward'
    contextual_backward = 'news-backward'

    if contextual_forward_path:
        contextual_forward = contextual_forward_path
    if contextual_backward_path:
        contextual_backward = contextual_backward_path

    if pooled:
        logger.info('Use PooledFlairEmbeddings with mean pooling')
        ContextualEmbeddings = partial(PooledFlairEmbeddings, pooling='mean')
    else:
        logger.info('Use FlairEmbeddings')
        ContextualEmbeddings = FlairEmbeddings

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings(word_embeddings),
        ContextualEmbeddings(contextual_forward),
        ContextualEmbeddings(contextual_backward),
    ]

    return embedding_types

def get_model(
        corpus: flair.data.Corpus, 
        pooled_contextual_embeddings: bool,
        contextual_forward_path: str = None,
        contextual_backward_path: str = None,
    ):

    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type = tag_type)

    embedding_types: List[TokenEmbeddings] = get_embeddings(
        pooled=pooled_contextual_embeddings,
        contextual_forward_path=contextual_forward_path,
        contextual_backward_path=contextual_backward_path
    )
    
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type)
    return tagger

def main():
    corpus = CorpusLoader().load_corpus(CORPUS_PATH['dummy'])
    tokenizer = TokenizerFactory()
    logger.info('Loaded corpus: {}'.format(corpus))

    model_dir = train_utils.model_dir(corpus.name, 'test')
    os.makedirs(model_dir, exist_ok = True)

    logger.info('Get sentences...')

    train_sents, train_docs = flair_utils.standoff_to_flair_sents(corpus.train, tokenizer)
    dev_sents, dev_docs = flair_utils.standoff_to_flair_sents(corpus.dev, tokenizer)
    test_sents, test_docs = flair_utils.standoff_to_flair_sents(corpus.test, tokenizer)
    
    logger.info('Train sentences: {}, Train docs: {}'.format(len(train_sents), len(train_docs)))
    logger.info('Dev sentences: {}, Dev docs: {}'.format(len(dev_sents), len(dev_docs)))
    logger.info('Test sentences: {}, Test docs: {}'.format(len(test_sents), len(test_docs)))

    flair_corpus = flair_utils.FilteredCorpus(train=train_sents,
                                              dev=dev_sents,
                                              test=test_sents,
                                              ignore_sentence=_ignore_sentence)
    logger.info(flair_corpus)

    logger.info('Train model...')

    tagger = get_model(
        flair_corpus,
        pooled_contextual_embeddings=False,
    )

    trainer = ModelTrainer(tagger, flair_corpus)
    trainer.train(join(model_dir, 'flair'),
                    max_epochs=5,
                    monitor_train=False,
                    train_with_dev=False,
                    embeddings_in_memory=False)

    if not False:
        # Model performance is judged by dev data, so we also pick the best performing model
        # according to the dev score to make our final predictions.
        tagger = SequenceTagger.load(join(model_dir, 'flair', 'best-model.pt'))
    else:
        # Training is stopped if train loss converges - here, we do not have a "best model" and
        # use the final model to make predictions.
        pass

    logger.info('Make predictions...')
    make_predictions(tagger, flair_corpus)

    train_utils.save_predictions(corpus_name=corpus.name, run_id='test',
                                 train=flair_utils.flair_sents_to_standoff(
                                     train_sents, train_docs),
                                 dev=flair_utils.flair_sents_to_standoff(
                                     dev_sents, dev_docs),
                                 test=flair_utils.flair_sents_to_standoff(test_sents, test_docs))



if __name__ == '__main__':
    main()