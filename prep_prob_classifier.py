from __future__ import absolute_import, division, unicode_literals

import os
import io
import copy
import logging
import numpy as np
from senteval.probing import PROBINGEval
from senteval.tools.validation import SplitClassifier

import torch

class PROBINGEvalWithKey(PROBINGEval):
    def __init__(self, task, task_path, seed=1111):
        super(PROBINGEvalWithKey,self).__init__(self, task, task_path, seed=1111)

    def run_with_tok(self, params, batcher):
        task_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size
        logging.info('Computing embeddings for train/dev/test')
        for key in self.task_data:
            # Sort to reduce padding
            sorted_data = sorted(zip(self.task_data[key]['X'],
                                     self.task_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.task_data[key]['X'], self.task_data[key]['y'] = map(list, zip(*sorted_data))
            self.task_data[key]['y_X'] = map(list, zip(*sorted_data))
            task_embed[key]['X'] = []
            for ii in range(0, len(self.task_data[key]['y']), bsize):
                batch_with_key = self.task_data[key]['y_X'][ii:ii + bsize]
                
                embeddings = batcher(params, batch_with_key)
                task_embed[key]['X'].append(embeddings)
            task_embed[key]['X'] = np.vstack(task_embed[key]['X'])
            task_embed[key]['y'] = np.array(self.task_data[key]['y'])
        logging.info('Computed embeddings')

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        if self.task == "WordContent" and params.classifier['nhid'] > 0:
            config_classifier = copy.deepcopy(config_classifier)
            config_classifier['classifier']['nhid'] = 0 
            print(params.classifier['nhid'])

        clf = SplitClassifier(X={'train': task_embed['train']['X'],
                                 'valid': task_embed['dev']['X'],
                                 'test': task_embed['test']['X']},
                              y={'train': task_embed['train']['y'],
                                 'valid': task_embed['dev']['y'],
                                 'test': task_embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : %.1f Test acc : %.1f for %s classification\n' % (devacc, testacc, self.task.upper()))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(task_embed['dev']['X']),
                'ntest': len(task_embed['test']['X'])}

class WordContentEval(PROBINGEvalWithKey):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'word_content.txt')
        # labels: 200 target words
        PROBINGEvalWithKey.__init__(self, 'WordContent', task_path, seed)

