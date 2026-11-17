
    
  
            st.warning(str(r['issue']))
    else:
        st.warning('This trace did not fit with the selected windows. See the issues table below.')

    with st.expander('Fit diagnostics for the selected trace'):
        if models:
            diagnostic = pd.DataFrame([
                {'Model':'Single straight line','Parameters':2,**models['linear']},
                {'Model':'Continuous two-line breakpoint','Parameters':4,**models['segmented']},
                {'Model':'Smooth exponential slope transition','Parameters':4,**models['transition']}
            ]).drop(columns=['predicted_A'],errors='ignore')
            st.dataframe(diagnostic[['Model','Parameters','n','r2','rmse_A','aicc']],
                         hide_index=True, use_container_width=True)
            st.caption('AICc is comparable only because all three models fit the same retained '
                       'time points. The best (lowest) AICc is not proof of a particular chemical mechanism. '
                       'Breakpoint-search AICc is approximate.')

st.subheader('Velocities versus [pyruvate]')
if summary['enzyme'].nunique() > 1:
    enzymes = sorted(summary['enzyme'].unique())
    target_enzyme = st.selectbox('Show enzyme/variant', enzymes)
else:
    target_enzyme = summary['enzyme'].iloc[0]
trace_group = valid[valid['enzyme']==target_enzyme].copy()
trace_group = trace_group[pd.to_numeric(trace_group['pyruvate_mM'],errors='coerce').notna()]

substrate_fits = {}
kinetics_fig = go.Figure()
for phase,col,color,label in [('v_early_uM_s','early_se_uM_s',BLUE,'Early-window v'),
                               ('v_late_uM_s','late_se_uM_s',ORANGE,'Late-window v')]:
    if len(trace_group):
        err = trace_group[col].fillna(0).to_numpy(float) if col in trace_group else None
        kinetics_fig.add_trace(go.Scatter(x=trace_group['pyruvate_mM'],
                        y=trace_group[phase],mode='markers',name=label,
                        marker={'color':color,'size':9},error_y=dict(type='data',array=err,visible=True),
                        text=trace_group['file'],hovertemplate='%{text}<br>S=%{x:.4g} mM<br>v=%{y:.4g} μM/s<extra></extra>'))
        try:
            fit=fit_substrate_inhibition(trace_group.pyruvate_mM.to_numpy(float),
                                        trace_group[phase].to_numpy(float))
            substrate_fits[phase]=fit
            smax=float(trace_group.pyruvate_mM.max())
            curve=np.linspace(0,smax,350)
            kinetics_fig.add_trace(go.Scatter(x=curve,
                      y=substrate_inhibition(curve,fit['vmax_uM_s'],fit['km_mM'],fit['ki_mM']),
                      name=f'{label} · inhibition fit',mode='lines',
                      line=dict(color=color, width=2)))
        except DataError:
            pass
kinetics_fig.update_layout(xaxis_title='[Pyruvate] (mM)',
                           yaxis_title='NADH consumption rate (μM/s)',height=410,
                           margin=dict(l=30,r=25,t=20,b=30),hovermode='closest')
st.plotly_chart(kinetics_fig,use_container_width=True)

ratio_fig=go.Figure()
if len(trace_group):
    ratio_fig.add_trace(go.Scatter(x=trace_group.pyruvate_mM,
                           y=trace_group.v_late_over_early,mode='markers+lines',
                           marker=dict(color=PURPLE,size=9),name='Late / early'))
ratio_fig.add_hline(y=1, line_dash='dash',line_color=GRAY)
ratio_fig.update_layout(xaxis_title='[Pyruvate] (mM)',yaxis_title='Late velocity / early velocity',
                        height=280,margin=dict(l=30,r=25,t=15,b=30))
st.plotly_chart(ratio_fig,use_container_width=True)

if substrate_fits:
    st.markdown('**Substrate-inhibition fits (separate apparent parameter sets)**')
    rows=[]
    for phase, fit in substrate_fits.items():
        rows.append({'Phase':phase.replace('v_','').replace('_uM_s',''),
                     'Vmax (μM/s)':fit['vmax_uM_s'],
                     'KM (mM)':fit['km_mM'],
                     'KI (mM)':fit['ki_mM'],
                     'R²':fit['r2'], 'Caution':fit['warning']})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.caption('Model: v = Vmax[S] / (KM + [S] + [S]²/KI). '
               'Because time-dependent phases need not be steady state, Vmax, KM, and KI '
               'here are descriptive *apparent* parameters, not necessarily microscopic constants.')
else:
    st.caption('To fit early- and late-phase substrate-inhibition curves, provide at least '
               'five different positive pyruvate concentrations with valid positive velocities.')

st.subheader('Results and downloads')
st.dataframe(summary,hide_index=True,use_container_width=True)

csv_content=summary.to_csv(index=False).encode('utf-8-sig')
col1,col2,col3=st.columns(3)
with col1:
    st.download_button('Download full velocity summary (CSV)',csv_content,
                       file_name='LDHA_two_phase_velocities.csv',mime='text/csv')
with col2:
    if models and selected['early'] and selected['late']:
        settings={'Selected trace':current,
                  'Extinction coefficient (mM^-1 cm^-1)':extinction,
                  'Pathlength (cm)':path_cm,
                  'Initial NADH (uM)':initial_nadh,
                  'Blank slope (A/s)':blank_slope,
                  'Dead time excluded (s)':dead_time_s}
        pdf=make_pdf(current,curr_df,models,selected['early'],selected['late'],
                     trace_group,substrate_fits,settings)
        st.download_button('Download 3-page PDF report',pdf,
                           file_name='LDHA_two_phase_report.pdf',mime='application/pdf')
with col3:
    model_rows=[]
    for name,fit in fits.items():
        if fit['models']:
            m=fit['models']
            for i,time_s in enumerate(m['t']):
                model_rows.append({'file':name,'time_s':time_s,'observed_A':m['A'][i],
                                  'single_A':m['linear']['predicted_A'][i],
                                  'segmented_A':m['segmented']['predicted_A'][i],
                                  'exponential_A':m['transition']['predicted_A'][i]})
    if model_rows:
        st.download_button('Download fitted time courses (CSV)',
                           pd.DataFrame(model_rows).to_csv(index=False).encode('utf-8-sig'),
                           file_name='LDHA_fitted_traces.csv',mime='text/csv')

with st.expander('Scientific limitations and suggested controls'):
    st.markdown('''
- Early/late slopes depend on the selected windows. A visible curve may not have two genuinely linear phases.
- An exponential slope transition is phenomenological. Its **k_obs** is not automatically an adduct formation constant.
- As NADH decreases and NAD⁺ builds up, the reaction environment changes; late-phase rates may be product- or substrate-dependent.
- Mixing artifacts and the instrumental dead time can bias early slopes. Compare mixing-only controls and replicate traces.
- Check the fraction of NADH consumed. If substantial, **neither late nor early slopes necessarily represent initial rates**.
- The software does not perform kinetic modeling of the complete ordered Bi–Bi reaction network or of covalent adduct formation.
- When possible, repeat runs with varied initial NADH and independent mixing/control experiments.
- Data are processed in the active Streamlit session and are not written to an app database. The user downloads results explicitly.
''')

