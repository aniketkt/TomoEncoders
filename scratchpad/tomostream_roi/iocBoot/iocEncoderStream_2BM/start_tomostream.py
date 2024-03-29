# This script creates an object of type TomoStream for doing tomography streaming reconstruction
# To run this script type the following:
#     python -i start_tomostream.py
# The -i is needed to keep Python running, otherwise it will create the object and exit
import sys
sys.path.append('/home/beams/TOMO/epics/synApps/support/tomostream_roi/tomostream')
from tomostream3d import EncoderStream
ts = EncoderStream(["../../db/tomoStream_settings.req","../../db/tomoStream_settings.req"], {"$(P)":"2bmb:", "$(R)":"EncoderStream:"})
