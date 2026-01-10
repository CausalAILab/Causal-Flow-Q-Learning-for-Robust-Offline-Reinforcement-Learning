from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.cfd_fql import Robust_FQLAgent
from agents.en_cfd_fql import Robust_Ensemble_FQLAgent

agents = dict(
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    robust_fql=Robust_FQLAgent,
    robust_en_fql=Robust_Ensemble_FQLAgent
)
