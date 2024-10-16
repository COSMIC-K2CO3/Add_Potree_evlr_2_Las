import Add_evlr
import Analyze_evlr

LsPath = "./LAS/"
PtPath = "./PoTree/"
OtPath = "./Output/"

Add_evlr.Add_evlr_to_pt(LsPath, PtPath, OtPath)
pt = Analyze_evlr.Potree('./Output/output_with_evlr.las')
position = pt.get_data_by_LOD(['position'],0)
print(position)