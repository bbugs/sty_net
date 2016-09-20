
from sqlalchemy import Column, Integer, PickleType, Numeric, Boolean, UniqueConstraint, String
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import create_engine

Base = declarative_base()


class Experiment(Base):
    __tablename__ = 'experiment'
    id = Column(Integer, primary_key=True)
    done = Column(Boolean)
    status = Column(String(20))
    time = Column(Numeric(asdecimal=False))
    best_val_f1 = Column(Numeric(asdecimal=False))
    train_f1_of_best_val = Column(Numeric(asdecimal=False))

    best_epoch = Column(Integer)
    priority = Column(Numeric(asdecimal=False))  # between 0 and 1. Initially randomly assigned

    reg = Column(Numeric(asdecimal=False))
    hidden_dim = Column(Integer)
    use_finetune_cnn = Column(Boolean)
    use_finetune_w2v = Column(Boolean)

    use_local = Column(Numeric(asdecimal=False))
    use_mil = Column(Boolean)

    use_global = Column(Numeric(asdecimal=False))
    global_margin = Column(Numeric(asdecimal=False))
    thrglobalscore = Column(Boolean)

    use_associat = Column(Numeric(asdecimal=False))

    update_rule = Column(String(20))
    learning_rate = Column(Numeric(asdecimal=False))

    UniqueConstraint(reg, hidden_dim, use_finetune_cnn, use_finetune_w2v,
                     use_local, use_mil,
                     use_global, global_margin, thrglobalscore,
                     use_associat,
                     update_rule, learning_rate)

    # attributes = Column(PickleType, unique=False)


# ##############
# insert at the end
# ##############

engine = create_engine('sqlite:///experiments.db')  # this creates the restaurant.db file

Base.metadata.create_all(engine)


# attributes or columns?
# """
# reg
# hidden_dim
# use_finetune_cnn
# use_finetune_w2v
#
# use_local
# use_mil
#
# use_global
# global_margin
# global_method
# thrglobalscore
#
# use_associat
#
# update_rule
# learning_rate
#
# """

