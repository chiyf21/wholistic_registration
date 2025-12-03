#%%
import nd2
import os
from glob import glob
from wholistic_registration.utils import IO
from wholistic_registration.core import main_function
from importlib import reload
reload(IO)
reload(main_function)


nd2folder = "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia_t23/exp0/anat/end/"

nd2files = glob(os.path.join(nd2folder, "*.nd2"))
nd2file = nd2files[0]
print(nd2file)

configfilePath = nd2file.replace(".nd2", ".toml")


metadata = IO.readMeta_new(nd2file)
config = main_function.DefineParams(inputFile=nd2file, configFile=configfilePath)
#%%