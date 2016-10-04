from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
import random
from net.multimodal.experiment_db.experiment_db_setup import Base, Experiment, setup_db


def populate_db(db_fname, n_exps):

    setup_db(db_fname)

    engine = create_engine(db_fname)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    hidden_dims = [500, 800, 1000, 1200]

    # Local Loss
    config = {}
    config['use_local'] = 1.
    config['use_global'] = 0.
    config['use_associat'] = 0.
    config['use_mil'] = False
    config['use_finetune_cnn'] = False
    config['use_finetune_w2v'] = False
    config['update_rule'] = 'sgd'
    config['done'] = False

    for i in range(n_exps):
        reg = 10 ** random.uniform(-10, -6)  # regularization
        lr = 10 ** random.uniform(-8, -2)  # learning rate
        hd = random.sample(hidden_dims, 1)[0]  # choose one element from hidden dims

        config['reg'] = reg
        config['learning_rate'] = lr
        config['hidden_dim'] = hd
        config['priority'] = np.round(np.random.uniform(0., 1.), 4)

        s = Experiment(**config)
        session.add(s)  # add row

    session.commit()

    session.close()

    engine.dispose()

    return