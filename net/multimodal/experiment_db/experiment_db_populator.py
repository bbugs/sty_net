from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pickle
import numpy as np

from cs231n.multimodal.experiment_db.experiment_db_setup import Base, Experiment
# from flask.ext.sqlalchemy import SQLAlchemy

engine = create_engine('sqlite:///experiments.db')

Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)

session = DBSession()

# Create the conditions to experiment with
configs = []

# regs = 10 ** np.array([i for i in range(-6, 2)])
regularizations = 10 ** np.random.uniform(-8, 3, size=8)
# learning_rates = 10 ** np.array([i for i in range(-8, 0)])
learning_rates = 10 ** np.random.uniform(-8, 0, size=9)
hidden_dims = [500, 800, 1000, 1200]

# use_local = [0., 1.]
# use_mil = [True]

# use_global = [0., 1.]


# use_associat = [0]

# config = {'reg': 0.1, 'hidden_dim': 78}

# Local Loss
config = {}
use_local = 1.
use_global = 0.
use_associat = 0.
use_mil = True
use_finetune_cnn = True
use_finetune_w2v = False
done = False
update_rule = 'sgd'
config['use_local'] = use_local
config['use_global'] = use_global
config['use_associat'] = use_associat
config['use_mil'] = use_mil
config['use_finetune_cnn'] = use_finetune_cnn
config['use_finetune_w2v'] = use_finetune_w2v
config['update_rule'] = update_rule
config['done'] = done
for reg in np.nditer(regularizations):
    for lr in np.nditer(learning_rates):
        for hd in hidden_dims:
            config['reg'] = reg
            config['learning_rate'] = lr
            config['hidden_dim'] = hd
            config['priority'] = np.round(np.random.uniform(0., 1.), 4)
            s = Experiment(**config)
            session.add(s)  # add row


regularizations = 10 ** np.random.uniform(-8, 3, size=8)
# learning_rates = 10 ** np.array([i for i in range(-8, 0)])
learning_rates = 10 ** np.random.uniform(-8, 0, size=9)
hidden_dims = [500, 800, 1000, 1200]

# Global Loss
config = {}
use_global = 1.
global_margin = 40.
thrglobalscore = True
use_local = 0.
use_associat = 0.
use_mil = False
use_finetune_cnn = True
use_finetune_w2v = False
done = False
update_rule = 'sgd'
config['use_local'] = use_local
config['use_global'] = use_global
config['use_associat'] = use_associat
config['use_mil'] = use_mil
config['use_finetune_cnn'] = use_finetune_cnn
config['use_finetune_w2v'] = use_finetune_w2v
config['update_rule'] = update_rule
config['global_margin'] = global_margin
config['thrglobalscore'] = thrglobalscore
config['done'] = done
for reg in np.nditer(regularizations):
    for lr in np.nditer(learning_rates):
        for hd in hidden_dims:
            config['reg'] = reg
            config['learning_rate'] = lr
            config['hidden_dim'] = hd
            config['priority'] = np.round(np.random.uniform(0., 1.), 4)
            s = Experiment(**config)
            session.add(s)  # add row


session.commit()

# config = {'a': 1, 'b': 2, 'c': 3}
# # Just set the attribute to save it
# s = Experiment(attributes=config, done=0)
# session.add(s)
#
# config = {'d': 1, 'e': 2, 'f': 3}
# # Just set the attribute to save it
# s = Experiment(attributes=config, done=0)
# session.add(s)


