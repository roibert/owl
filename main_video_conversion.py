from src.video_processing.video_to_single_frame import video_to_single_frame

path_pc_exp1 = r"ADD PATH HERE TO VIDEO FILES"
path_pc_exp2 = r"ADD PATH HERE TO VIDEO FILES"
path_pc_exp3 = r"ADD PATH HERE TO VIDEO FILES"

output_path_pc_exp1 = r"ADD PATH HERE TO FRAME OUTPUT FOLDER"
output_path_pc_exp2 = r"ADD PATH HERE TO FRAME OUTPUT FOLDER"
output_path_pc_exp3 = r"ADD PATH HERE TO FRAME OUTPUT FOLDER"

cropping_exp1 = [170, 360, 300, 400] # example cropping coordinates
cropping_exp2 = [250, 420, 400, 500] # example cropping coordinates
cropping_exp3 = [350, 450, 200, 350] # example cropping coordinates

video_to_single_frame(path_pc_exp1, "r8", output_path_pc_exp1, cropping_exp1)
video_to_single_frame(path_pc_exp2, "r8", output_path_pc_exp2, cropping_exp2)
video_to_single_frame(path_pc_exp3, "r8", output_path_pc_exp3, cropping_exp3)
