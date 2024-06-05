import control_function as cf
import json_control as jc

json_path = "code/skeleton_reference_control/control.json"
video_key_list = ["1.의자에 앉아 물병 들어올리기", "2.의자에 앉아 물병 옆으로 들어올리기", "3.의자에 앉아 물병 머리 뒤로 들어올리기",
                  "4.의자에 앉아 팔 펴서 물병 들어올리기", "5.의자에 앉았다 일어서기", "6.서서 무릎 들기", 
                  "7.서서 무릎 굽히기", "8.서서 무릎 뒤로 뻗기", "9.서서 다리 옆으로 들기"]

for i in video_key_list:
    video_key = video_key_list[i]
    config = jc.from_json(json_path, video_key)
    cf.set_minmax_range(**config.to_dict())