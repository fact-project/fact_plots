
#dict(bins=100, log=True, func="np.log10", range=[-10, 40] )
default_plots = {
'Alpha' : None,
'Alpha_Off_1' : None,
'Alpha_Off_2' : None,
'Alpha_Off_3' : None,
'Alpha_Off_4' : None,
'Alpha_Off_5' : None,
'AzPointing' : False,
'AzSourceCalc' : None,
'AzTracking' : False,
'COGx' : dict(bins=100, range=[-200, 250], log=True, func=False, xUnit="mm"),
'COGy' : dict(bins=100, range=[-200, 250], log=True, func=False, xUnit="mm"),
'ConcCore' : dict(bins=100, range=[0.0, 100.0], log=True, func="100*", xUnit="%"),
'concCOG' : dict(bins=100, range=[0.0, 100.0], log=True, func="100*", xUnit="%"),
'Concentration_onePixel' : dict(bins=100, range=[0.0, 100.0], log=True, func="100*", xUnit="%"),
'Concentration_twoPixel' : dict(bins=100, range=[0.0, 100.0], log=True, func="100*", xUnit="%"),
'CosDeltaAlpha' : dict(bins=100, range=[-1.0, 2], log=False, func=False, xUnit="a.u."),
'CosDeltaAlpha_Off_1' : dict(bins=100, range=[-1.0, 2], log=False, func=False, xUnit="a.u."),
'CosDeltaAlpha_Off_2' : dict(bins=100, range=[-1.0, 2], log=False, func=False, xUnit="a.u."),
'CosDeltaAlpha_Off_3' : dict(bins=100, range=[-1.0, 2], log=False, func=False, xUnit="a.u."),
'CosDeltaAlpha_Off_4' : dict(bins=100, range=[-1.0, 2], log=False, func=False, xUnit="a.u."),
'CosDeltaAlpha_Off_5' : dict(bins=100, range=[-1.0, 2], log=False, func=False, xUnit="a.u."),
'Delta' : dict(bins=130, range=[-2, 2], log=False, func=False, xUnit="a.u." ),
'Disp' : dict(bins=130, range=[0, 130], log=True, func=False, xUnit="a.u." ),
'Distance' : dict(bins=100, range=[0.0, 300.0], log=True, func=False, xUnit="mm" ),
'Distance_Off_1' : dict(bins=100, range=[0.0, 300.0], log=True, func=False, xUnit="mm" ),
'Distance_Off_2' : dict(bins=100, range=[0.0, 300.0], log=True, func=False, xUnit="mm" ),
'Distance_Off_3' : dict(bins=100, range=[0.0, 300.0], log=True, func=False, xUnit="mm" ),
'Distance_Off_4' : dict(bins=100, range=[0.0, 300.0], log=True, func=False, xUnit="mm" ),
'Distance_Off_5' : dict(bins=100, range=[0.0, 300.0], log=True, func=False, xUnit="mm" ),
'EventNum' : dict(bins=100, range=[0.0, 15e4], log=True, func=False, xUnit="a.u." ),
'Leakage' : dict(bins=100, range=[0.0, 100.0], log=True, func="100*", xUnit="%" ),
'Leakage2' : dict(bins=100, range=[0.0, 100.0], log=True, func="100*", xUnit="%" ),
'Length' : dict(bins=100, range=[0.0, 100.0], log=True, func=False, xUnit="mm" ),
'M3Long' : dict(bins=100, range=[-1.5, 1.5], log=False, func=False, xUnit="a.u." ),
'M3Trans' : dict(bins=100, range=[-1.5, 1.5], log=False, func=False, xUnit="a.u." ),
'M4Long' : dict(bins=100, range=[0, 5], log=False, func=False, xUnit="a.u." ),
'M4Trans' : dict(bins=100, range=[0, 5], log=False, func=False, xUnit="a.u." ),
'NPIX' : False,
'NROI' : False,
'Size' : dict(bins=150, range=[1.0, 5.3], log=True, func="np.log10", xUnit="ph.e." ),
'Slope_long' : dict(bins=100, range=[-1, 1], log=False, func=False, xUnit="a.u." ),
'Slope_spread' : dict(bins=100, range=[0, 10], log=False, func=False, xUnit="a.u." ),
'Slope_spread_weighted' : dict(bins=100, range=[0, 10], log=False, func=False, xUnit="a.u." ),
'Slope_trans' : dict(bins=100, range=[-0.6, 0.6], log=False, func=False, xUnit="a.u." ),
'Theta' : dict(bins=100, range=[0, 300], log=False, func=False, xUnit="deg" ),
'Theta_Off_1' : dict(bins=100, range=[0, 300], log=False, func=False, xUnit="deg" ),
'Theta_Off_2' : dict(bins=100, range=[0, 300], log=False, func=False, xUnit="deg" ),
'Theta_Off_3' : dict(bins=100, range=[0, 300], log=False, func=False, xUnit="deg" ),
'Theta_Off_4' : dict(bins=100, range=[0, 300], log=False, func=False, xUnit="deg" ),
'Theta_Off_5' : dict(bins=100, range=[0, 300], log=False, func=False, xUnit="deg" ),
'Timespread' : dict(bins=100, range=[-1, 20], log=True, func="0.5*", xUnit="ns" ),
'Timespread_weighted' : dict(bins=100, range=[-1, 20], log=True, func="0.5*", xUnit="ns" ),
'Width' : dict(bins=150, range=[0, 100], log=True, func=False, xUnit="mm" ),
'ZdPointing' : dict(bins=30, range=[-5, 32], log=False, func=False, xUnit="deg" ),
'ZdSourceCalc' : dict(bins=30, range=[-5, 32], log=False, func=False, xUnit="deg" ),
'ZdTracking' : dict(bins=30, range=[-5, 32], log=False, func=False, xUnit="deg" ),
'arrTimePedestal_kurtosis' : dict(bins=100, range=[-4, 12], log=True, func=False, xUnit="a.u." ),
'arrTimePedestal_max' : dict(bins=80, range=[45, 51], log=False, func="0.5*", xUnit="ns" ),
'arrTimePedestal_mean' : dict(bins=80, range=[20, 35], log=False, func="0.5*", xUnit="ns" ),
'arrTimePedestal_median' : dict(bins=80, range=[17, 35], log=False, func="0.5*", xUnit="ns" ),
'arrTimePedestal_min' : dict(bins=80, range=[0, 5], log=False, func="0.5*", xUnit="ns" ),
'arrTimePedestal_p25' : dict(bins=80, range=[10, 25], log=False, func="0.5*", xUnit="ns" ),
'arrTimePedestal_p75' : dict(bins=80, range=[30, 45], log=False, func="0.5*", xUnit="ns" ),
'arrTimePedestal_skewness' : dict(bins=100, range=[-0.5, 0.4], log=False, func=False, xUnit="a.u." ),
'arrTimePedestal_variance' : dict(bins=100, range=[16, 30], log=False, func="np.sqrt", xUnit="a.u." ),
'arrTimePosShower_max' : dict(bins=70, range=[10, 100], log=True, func=False, xUnit="slice" ),
'arrTimePosShower_mean' : dict(bins=50, range=[30, 90], log=False, func=False, xUnit="slice" ),
'arrTimePosShower_min' : dict(bins=40, range=[25, 65], log=False, func=False, xUnit="slice" ),
'arrTimePosShower_skewness' : dict(bins=100, range=[-10, 7], log=True, func=False, xUnit="slice" ),
'arrTimePosShower_kurtosis' : dict(bins=100, range=[-4, 20], log=True, func=False, xUnit="slice" ),
'arrTimePosShower_variance' : dict(bins=100, range=[0, 20], log=True, func="np.sqrt", xUnit="slice" ),
'arrTimeShower_kurtosis' : dict(bins=100, range=[-5, 15], log=False, func=False, xUnit="a.u." ),
'arrTimeShower_max' : dict(bins=100, range=[15, 55], log=False, func="0.5*", xUnit="ns" ),
'arrTimeShower_mean' : dict(bins=100, range=[15, 45], log=False, func="0.5*", xUnit="ns" ),
'arrTimeShower_min' : dict(bins=100, range=[15, 35], log=False, func="0.5*", xUnit="ns" ),
'arrTimeShower_skewness' : dict(bins=100, range=[-5, 5], log=False, func=False, xUnit="a.u." ),
'arrTimeShower_variance' : dict(bins=100, range=[0, 20], log=True, func="np.sqrt", xUnit="a.u." ),
'arrivalTimeMean' : dict(bins=100, range=[20, 40], log=False, func="0.5*", xUnit="ns" ),
'm3l' : dict(bins=100, range=[-600, 600], log=True, func=False, xUnit="a.u." ),
'm3t' : dict(bins=100, range=[-600, 600], log=True, func=False, xUnit="a.u." ),
'maxPosShower_kurtosis' : dict(bins=100, range=[-5, 20], log=False, func=False, xUnit="a.u." ),
'maxPosShower_max' : dict(bins=60, range=[40, 100], log=False, func=False, xUnit="a.u." ),
'maxPosShower_mean' : dict(bins=60, range=[40, 100], log=False, func=False, xUnit="a.u." ),
'maxPosShower_min' : dict(bins=50, range=[30, 80], log=False, func=False, xUnit="a.u." ),
'maxPosShower_skewness' : dict(bins=50, range=[-3, 5], log=False, func=False, xUnit="a.u." ),
'maxPosShower_variance' : dict(bins=100, range=[0, 20], log=True, func="np.sqrt", xUnit="a.u." ),
'maxSlopesPedestal_kurtosis' : dict(bins=100, range=[0, 60], log=True, func=None, xUnit="mV/slices" ),
'maxSlopesPedestal_max' : dict(bins=100, range=[0, 25], log=False, func=False, xUnit="mV/slices" ),
'maxSlopesPedestal_mean' : dict(bins=100, range=[1, 2.5], log=False, func=False, xUnit="mV/slices" ),
'maxSlopesPedestal_median' : dict(bins=100, range=[1, 2.5], log=False, func=False, xUnit="mV/slices" ),
'maxSlopesPedestal_min' : dict(bins=100, range=[-6, 2], log=False, func=False, xUnit="mV/slices" ),
'maxSlopesPedestal_p25' : dict(bins=100, range=[0, 2], log=False, func=False, xUnit="mV/slices" ),
'maxSlopesPedestal_p75' : dict(bins=100, range=[1.5, 3.5], log=False, func=False, xUnit="mV/slices" ),
'maxSlopesPedestal_skewness' : dict(bins=100, range=[0, 8], log=True, func=False, xUnit="mV/slices" ),
'maxSlopesPedestal_variance' : dict(bins=100, range=[0.9, 1.9], log=False, func="np.sqrt", xUnit="mV/slices" ),
'maxSlopesPosShower_kurtosis' : dict(bins=100, range=[-10, 20], log=False, func=False, xUnit="a.u." ),
'maxSlopesPosShower_max' : dict(bins=100, range=[0, 1000], log=True, func=False, xUnit="slices" ),
'maxSlopesPosShower_mean' : dict(bins=100, range=[0, 300], log=True, func=False, xUnit="slices" ),
'maxSlopesPosShower_min' : dict(bins=40, range=[0, 40], log=False, func=False, xUnit="slices" ),
'maxSlopesPosShower_skewness' : dict(bins=60, range=[-3, 5], log=False, func=False, xUnit="a.u." ),
'maxSlopesPosShower_variance' : dict(bins=100, range=[-1, 2500], log=True, func="np.sqrt", xUnit="a.u." ),
'maxSlopesShower_kurtosis' : dict(bins=100, range=[-5, 20], log=False, func=False, xUnit="a.u." ),
'maxSlopesShower_max' : dict(bins=100, range=[0, 400], log=True, func=False, xUnit="mV/slices" ),
'maxSlopesShower_mean' : dict(bins=100, range=[0, 300], log=True, func=False, xUnit="mV/slices" ),
'maxSlopesShower_min' : dict(bins=100, range=[-10, 20], log=True, func=False, xUnit="mV/slices" ),
'maxSlopesShower_skewness' : dict(bins=100, range=[-3, 5], log=False, func=False, xUnit="a.u." ),
'maxSlopesShower_variance' : dict(bins=100, range=[-1, 600], log=True, func="np.sqrt", xUnit="a.u." ),
'numIslands' : dict(bins=10, range=[0, 25], log=True, func=False, xUnit="a.u." ),
'numPixelInPedestal' : dict(bins=360, range=[0, 1440], log=True, func=False, xUnit="a.u." ),
'numPixelInShower' : dict(bins=360, range=[0, 1440], log=True, func=False, xUnit="a.u." ),
'ped_mean_kurtosis' : dict(bins=100, range=[-1, 15], log=False, func=False, xUnit="a.u." ),
'ped_mean_max' : dict(bins=100, range=[0, 8000], log=True, func=False, xUnit="a.u." ),
'ped_mean_mean' : dict(bins=100, range=[-400, 6000], log=True, func=False, xUnit="a.u." ),
'ped_mean_median' : dict(bins=100, range=[-50, 6000], log=True, func=False, xUnit="a.u." ),
'ped_mean_min' : dict(bins=100, range=[-4000, 3000], log=True, func=False, xUnit="a.u." ),
'ped_mean_p25' : dict(bins=100, range=[-400, 2000], log=True, func=False, xUnit="a.u." ),
'ped_mean_p75' : dict(bins=100, range=[-200, 2000], log=True, func=False, xUnit="a.u." ),
'ped_mean_skewness' : dict(bins=100, range=[0, 2.0], log=False, func=False, xUnit="a.u." ),
'ped_mean_variance' : dict(bins=100, range=[100, 1000], log=True, func="np.sqrt", xUnit="a.u." ),
'ped_median_kurtosis' : dict(bins=100, range=[0, 40], log=False, func=None, xUnit="a.u." ),
'ped_median_max' : dict(bins=100, range=[0, 1200], log=False, func=False, xUnit="a.u." ),
'ped_median_mean' : dict(bins=100, range=[-400, -50], log=False, func=False, xUnit="a.u." ),
'ped_median_median' : dict(bins=100, range=[-400, -50], log=False, func=False, xUnit="a.u." ),
'ped_median_min' : dict(bins=100, range=[-2500, 0], log=False, func=False, xUnit="a.u." ),
'ped_median_p25' : dict(bins=100, range=[-500, 0], log=False, func=False, xUnit="a.u." ),
'ped_median_p75' : dict(bins=100, range=[-350, 0], log=False, func=False, xUnit="a.u." ),
'ped_median_skewness' : dict(bins=100, range=[-4, 4], log=False, func=False, xUnit="a.u." ),
'ped_median_variance' : dict(bins=100, range=[50, 160], log=False, func="np.sqrt", xUnit="a.u." ),
'ped_std_kurtosis' : dict(bins=100, range=[0, 100], log=True, func=False, xUnit="a.u." ),
'ped_std_max' : dict(bins=100, range=[0, 7000], log=True, func=False, xUnit="a.u." ),
'ped_std_mean' : dict(bins=100, range=[0, 7000], log=True, func=False, xUnit="a.u." ),
'ped_std_median' : dict(bins=100, range=[100, 7000], log=True, func=False, xUnit="a.u." ),
'ped_std_min' : dict(bins=100, range=[-10, 6000], log=True, func=False, xUnit="a.u." ),
'ped_std_p25' : dict(bins=100, range=[50, 4000], log=True, func=False, xUnit="a.u." ),
'ped_std_p75' : dict(bins=100, range=[150, 5000], log=True, func=False, xUnit="a.u." ),
'ped_std_skewness' : dict(bins=100, range=[0, 5], log=False, func=False, xUnit="a.u." ),
'ped_std_variance' : dict(bins=100, range=[60, 1000], log=True, func="np.sqrt", xUnit="a.u." ),
'ped_sum_kurtosis' : dict(bins=100, range=[-1, 100], log=True, func=False, xUnit="a.u." ),
'ped_sum_max' : dict(bins=100, range=[0, 5e4], log=True, func=False, xUnit="a.u." ),
'ped_sum_mean' : dict(bins=100, range=[-2000, 4000], log=True, func=False, xUnit="a.u." ),
'ped_sum_median' : dict(bins=100, range=[-2000, 2000], log=True, func=False, xUnit="a.u." ),
'ped_sum_min' : dict(bins=100, range=[-12000, 0], log=True, func=False, xUnit="a.u." ),
'ped_sum_p25' : dict(bins=100, range=[-2000, 500], log=True, func=False, xUnit="a.u." ),
'ped_sum_p75' : dict(bins=100, range=[-1000, 4000], log=True, func=False, xUnit="a.u." ),
'ped_sum_skewness' : dict(bins=100, range=[-0.5, 2.5], log=True, func=False, xUnit="a.u." ),
'ped_sum_variance' : dict(bins=100, range=[700, 1500], log=False, func="np.sqrt", xUnit="a.u." ),
'ped_var_kurtosis' : dict(bins=100, range=[0, 30], log=False, func="np.sqrt", xUnit="a.u." ),
'ped_var_max' : dict(bins=100, range=[500, 2500], log=False, func="np.sqrt", xUnit="a.u." ),
'ped_var_mean' : dict(bins=100, range=[160, 260], log=False, func="np.sqrt", xUnit="a.u." ),
'ped_var_median' : dict(bins=100, range=[120, 220], log=False, func="np.sqrt", xUnit="a.u." ),
'ped_var_min' : dict(bins=100, range=[0, 35], log=False, func="np.sqrt", xUnit="a.u." ),
'ped_var_p25' : dict(bins=100, range=[80, 150], log=False, func="np.sqrt", xUnit="a.u." ),
'ped_var_p75' : dict(bins=100, range=[180, 300], log=False, func="np.sqrt", xUnit="a.u." ),
'ped_var_skewness' : dict(bins=100, range=[0, 8], log=True, func="np.sqrt", xUnit="a.u." ),
'ped_var_variance' : dict(bins=100, range=[1e5, 1e7], log=True, func="np.sqrt", xUnit="a.u." ),
'pedestalSize' : dict(bins=100, range=[0, 6500], log=False, func=False, xUnit="ph.e." ),
'pedestalTimespread' : dict(bins=100, range=[0, 16], log=False, func="0.5*", xUnit="ns" ),
'phChargePedestal_kurtosis' : dict(bins=100, range=[-1, 12], log=False, func=False, xUnit="a.u." ),
'phChargePedestal_max' : dict(bins=100, range=[0, 30], log=False, func=False, xUnit="ph.e." ),
'phChargePedestal_mean' : dict(bins=100, range=[-0.5, 2], log=False, func=False, xUnit="ph.e." ),
'phChargePedestal_median' : dict(bins=100, range=[-1, 2], log=False, func=False, xUnit="ph.e." ),
'phChargePedestal_min' : dict(bins=100, range=[-6, 1], log=False, func=False, xUnit="ph.e." ),
'phChargePedestal_p25' : dict(bins=100, range=[-1.5, 1], log=False, func=False, xUnit="ph.e." ),
'phChargePedestal_p75' : dict(bins=100, range=[0, 3], log=False, func=False, xUnit="ph.e." ),
'phChargePedestal_skewness' : dict(bins=100, range=[0, 2.5], log=False, func=False, xUnit="a.u." ),
'phChargePedestal_variance' : dict(bins=100, range=[1.0, 2.0], log=False, func="np.sqrt", xUnit="a.u." ),
'phChargeShower_kurtosis' : dict(bins=100, range=[-5, 15], log=False, func=False, xUnit="a.u." ),
'phChargeShower_max' : dict(bins=100, range=[0, 200], log=False, func=False, xUnit="ph.e." ),
'phChargeShower_mean' : dict(bins=100, range=[0, 40], log=False, func=False, xUnit="ph.e." ),
'phChargeShower_min' : dict(bins=100, range=[2, 8], log=False, func=False, xUnit="ph.e." ),
'phChargeShower_skewness' : dict(bins=100, range=[-3, 5], log=False, func=False, xUnit="a.u." ),
'phChargeShower_variance' : dict(bins=100, range=[0, 50], log=False, func="np.sqrt", xUnit="ph.e." ),
'photonchargeMean' : dict(bins=100, range=[-1, 5], log=False, func=False, xUnit="ph.e." ),
'sourcePosition' : False,
'timeGradientIntercept' : False,
'timeGradientIntercept_err' : False,
'timeGradientSSE' : False,
'timeGradientSlope' : False,
'timeGradientSlope_err' : False,
'TriggerType': False,
}