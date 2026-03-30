# import SessionData
# import compute_metrics, plot_functions, export_metrics
from stringpullkit.analysis import SessionData, compute_metrics, plot_functions, export_metrics


def run_analysis(video_path=None, dlc_paths=None, save_dir=None, fps=120, session_id="", total_frames=0, likelihood_threshold=0.6,
                        smoothing_window=25, smoothing_poly=2, scale_factor=None, height=None, generate_plot=True, show_plot=False):

    session = SessionData.SessionData(video_path=video_path, dlc_paths=dlc_paths, save_dir=save_dir, fps=fps, session_id=session_id,
                 total_frames=total_frames, likelihood_threshold=likelihood_threshold, smoothing_window=smoothing_window, 
                 smoothing_poly=smoothing_poly,  scale_factor=scale_factor, height=height, show_plot=show_plot)
    
    print("Loading session data...")
    session.load_data()
    
    print("Cleaning session data...")
    session.clean_data()
    
    print("Computing session metrics...")
    compute_metrics.compute_all_metrics(session)
    
    print("Saving session metrics...")
    export_metrics.save_all_metrics(session)
    
    if generate_plot:
      print("Plotting session metrcs...")
      plot_functions.plot_all_metrics(session)
    
    print("Analysis complete.")


# #nose_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Flx_Cohort_1\Control\Disease_Onset_Flx_Cohort_1\DLC_Analysis\WT\NC29L3A1_onset_TRIAL1.2_1\NC29L3A1_onset_TRIAL1.2_1DLC_resnet50_StringPull_Nose2Jun3shuffle1_1030000.csv"
# hands_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Flx_Cohort_1\Control\Disease_Onset_Flx_Cohort_1\DLC_Analysis\WT\NC29L3A1_onset_TRIAL1.2_1\NC29L3A1_onset_TRIAL1.2_1DLC_resnet50_StringPull_Hands2Jun4shuffle1_1030000.csv"
# string_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Flx_Cohort_1\Control\Disease_Onset_Flx_Cohort_1\DLC_Analysis\WT\NC29L3A1_onset_TRIAL1.2_1\NC29L3A1_onset_TRIAL1.2_1DLC_resnet50_StringPull_String3Jul3shuffle1_1030000.csv"
# feet_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Flx_Cohort_1\Control\Disease_Onset_Flx_Cohort_1\DLC_Analysis\WT\NC29L3A1_onset_TRIAL1.2_1\NC29L3A1_onset_TRIAL1.2_1DLC_resnet50_StringPull_FeetJun5shuffle1_1030000.csv"
# ears_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\Flx_Cohort_1\Control\Disease_Onset_Flx_Cohort_1\DLC_Analysis\WT\NC29L3A1_onset_TRIAL1.2_1\NC29L3A1_onset_TRIAL1.2_1DLC_resnet50_StringPull_Ears2Jun6shuffle1_1030000.csv"

# #save_dir = r"C:\Users\USER\Desktop\Watt Lab\String Pull\NC29L3A1_onset_TRIAL1.2_1_REPC1WT_refactor"
# save_dir = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\test_data\analysis_output"
# arms_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\test_data\dlc_output\NC29L3A1_onset_1.2_1DLC_resnet50_StringPull_ArmsOct31shuffle1_1030000.csv"
# ears_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\test_data\dlc_output\NC29L3A1_onset_1.2_1DLC_resnet50_StringPull_Ears2Jun6shuffle1_1030000.csv"
# feet_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\test_data\dlc_output\NC29L3A1_onset_1.2_1DLC_resnet50_StringPull_FeetJun5shuffle1_1030000.csv"
# hands_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\test_data\dlc_output\NC29L3A1_onset_1.2_1DLC_resnet50_StringPull_Hands2Jun4shuffle1_1030000.csv"
# string_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\test_data\dlc_output\NC29L3A1_onset_1.2_1DLC_resnet50_StringPull_String3Jul3shuffle1_1030000.csv"
# torso_filepath = r"\\m40-gci.sci.mcgill.ca\Watt-Projects\Rana-Juliana\String_Pull\test_data\dlc_output\NC29L3A1_onset_1.2_1DLC_resnet50_StringPull_TorsoNov3shuffle1_1030000.csv"

# dlc_csv_paths = {"Arms": arms_filepath, "Ears": ears_filepath, "Feet": feet_filepath, "Hands": hands_filepath, "String": string_filepath, "Torso": torso_filepath}
# #
# run_analysis(video_path=None, dlc_paths=dlc_csv_paths, fps=120, scale_factor=1/10, show_plot=False, session_id="MouseButts", height=1416, save_dir=save_dir)
# # print("Done")
