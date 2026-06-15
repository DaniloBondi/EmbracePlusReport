from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import io
import re
import zipfile
import tempfile
import os
from collections import defaultdict
from pathlib import Path

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("zip_file", "Upload ZIP file", accept=[".zip"], multiple=False),
        ui.input_action_button("process", "Process Data", class_="btn-primary"),
        ui.download_button("download_report", "Download PDF Report"),
        width=300
    ),
    ui.card(
        ui.card_header("Processing Status"),
        ui.output_text_verbatim("status")
    ),
    ui.card(
        ui.card_header("Summary Statistics"),
        ui.output_ui("summary_display")
    )
)

def server(input, output, session):
    # Reactive values to store processed data
    processed_data = reactive.value(None)
    
    @reactive.effect
    @reactive.event(input.process)
    def _process_files():
        zip_file = input.zip_file()
        if not zip_file:
            ui.notification_show("Please upload a ZIP file first", type="error")
            return
        
        try:
            # Process ZIP file
            result = process_zip_file(zip_file[0])
            processed_data.set(result)
            ui.notification_show("Data processed successfully!", type="message")
        except Exception as e:
            ui.notification_show(f"Error processing files: {str(e)}", type="error")
    
    @output
    @render.text
    def status():
        zip_file = input.zip_file()
        if not zip_file:
            return "No ZIP file uploaded yet."
        
        data = processed_data.get()
        if data is None:
            return f"ZIP file uploaded. Click 'Process Data' to begin."
        
        return f"Processing complete!\nDates identified: {data['num_dates']}\nValid days: {data['num_valid_days']}\nValid nights: {data['num_valid_nights']}"
    
    @output
    @render.ui
    def summary_display():
        data = processed_data.get()
        if data is None or 'summary_metrics' not in data:
            return ui.p("Process data to see summary statistics")
        
        metrics = data['summary_metrics']
        
        # Create HTML table
        rows = []
        for key, value in metrics.items():
            rows.append(f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>")
        
        html_table = f"""
        <table class="table table-striped">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        """
        
        return ui.HTML(html_table)
    
    @render.download(filename="activity_report.pdf")
    def download_report():
        data = processed_data.get()
        if data is None:
            yield b""
            return
        
        # Create PDF
        buffer = io.BytesIO()
        create_pdf_report(data, buffer)
        buffer.seek(0)
        yield buffer.getvalue()

def process_zip_file(zip_file_info):
    """Extract and process CSV files from ZIP archive"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file_info["datapath"], 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        csv_files = []
        file_pattern = re.compile(r'.*_(\d{4}-\d{2}-\d{2})_(.+)\.csv$')
        
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                if filename.endswith('.csv'):
                    match = file_pattern.search(filename)
                    if match:
                        file_path = os.path.join(root, filename)
                        csv_files.append({
                            "name": filename,
                            "datapath": file_path,
                            "date": match.group(1),
                            "var_type": match.group(2)
                        })
        
        return process_csv_files(csv_files)

def process_csv_files(files):
    """Process uploaded CSV files and return analysis results"""
    organized_files = defaultdict(dict)
    for file_info in files:
        organized_files[file_info["date"]][file_info["var_type"]] = file_info
    
    daily_merged_list = []
    for file_date in sorted(organized_files.keys()):
        daily_vars = []
        for var_type, file_info in organized_files[file_date].items():
            try:
                df = pd.read_csv(file_info["datapath"], sep=None, engine='python')
                if 'timestamp_iso' in df.columns:
                    df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'], errors='coerce')
                    
                    if 'missing_value_reason' in df.columns:
                        df['missing_value_reason'] = df['missing_value_reason'].astype(str)
                    if 'participant_full_id' in df.columns:
                        df['participant_full_id'] = df['participant_full_id'].astype(str)
                    if 'timestamp_unix' in df.columns:
                        df['timestamp_unix'] = pd.to_numeric(df['timestamp_unix'], errors='coerce')
                    
                    cols_to_rename = [c for c in df.columns if c not in ['timestamp_iso', 'timestamp_unix', 'participant_full_id', 'missing_value_reason']]
                    if len(cols_to_rename) == 1:
                        df = df.rename(columns={cols_to_rename[0]: var_type})
                    daily_vars.append(df)
            except Exception:
                pass
        
        if daily_vars:
            merged_day_df = daily_vars[0].copy()
            for next_df in daily_vars[1:]:
                common_cols = ['timestamp_iso', 'timestamp_unix', 'participant_full_id', 'missing_value_reason']
                common_cols = [c for c in common_cols if c in merged_day_df.columns and c in next_df.columns]
                
                if common_cols:
                    merged_day_df = pd.merge(merged_day_df, next_df, on=common_cols, how='outer')
                else:
                    merged_day_df = pd.concat([merged_day_df, next_df], axis=1)
            
            merged_day_df['date_label'] = file_date
            daily_merged_list.append(merged_day_df)
    
    if not daily_merged_list:
        return None

    final_merged_df = pd.concat(daily_merged_list, ignore_index=True)
    final_merged_df['timestamp_iso'] = pd.to_datetime(final_merged_df['timestamp_iso'], errors='coerce')
    
    if 'participant_full_id' in final_merged_df.columns:
        final_merged_df['participant_full_id'] = final_merged_df['participant_full_id'].ffill().bfill()
    
    final_merged_df = final_merged_df.sort_values('timestamp_iso').drop_duplicates(subset=['timestamp_iso'])
    final_merged_df = final_merged_df.set_index('timestamp_iso')
    
    full_range = pd.date_range(start=final_merged_df.index.min(), end=final_merged_df.index.max(), freq='1min')
    final_merged_df = final_merged_df.reindex(full_range)
    final_merged_df.index.name = 'timestamp_iso'
    final_merged_df = final_merged_df.reset_index()
    
    numeric_columns = ['step-counts', 'eda', 'pulse-rate', 'respiratory-rate', 'temperature', 'met', 'prv']
    for col in numeric_columns:
        if col in final_merged_df.columns:
            final_merged_df[col] = pd.to_numeric(final_merged_df[col], errors='coerce')
    
    if 'sleep-detection' in final_merged_df.columns:
        final_merged_df['sleep-detection'] = pd.to_numeric(final_merged_df['sleep-detection'], errors='coerce')
    
    # Inizializzazione variabili per evitare il bug del NameError
    valid_days = []
    missing_metrics = {}
    minuti_non_registrati_tot = 0
    ore_non_registrate_tot = 0.0
    percentuale_non_registrata = 0.0

    if 'step-counts' in final_merged_df.columns and 'missing_value_reason' in final_merged_df.columns:
        quality_check = final_merged_df.groupby('date_label')['missing_value_reason'].apply(
            lambda x: (x == 'device_not_recording').sum()
        ).reset_index()
        quality_check.columns = ['Data', 'Minuti_Senza_Registrazione']
        
        soglia_errore = 1440 * 0.1
        valid_days = quality_check[quality_check['Minuti_Senza_Registrazione'] <= soglia_errore]['Data'].tolist()
        quality_check_valid = quality_check[quality_check['Data'].isin(valid_days)]
        
        if not quality_check_valid.empty:
            minuti_non_registrati_tot = quality_check_valid['Minuti_Senza_Registrazione'].sum()
            ore_non_registrate_tot = minuti_non_registrati_tot / 60
            giorni_validi = len(quality_check_valid)
            minuti_attesi_validi = giorni_validi * 1440
            # CORRETTO: Variabile in italiano 'minuti_attesi_validi'
            percentuale_non_registrata = (minuti_non_registrati_tot / minuti_attesi_validi * 100) if minuti_attesi_validi > 0 else 0.0

        missing_metrics = {
            'Tempo senza registrazione (min, giorni validi)': f'{minuti_non_registrati_tot:.0f}',
            'Tempo senza registrazione (ore, giorni validi)': f'{ore_non_registrate_tot:.2f}',
            '% tempo senza registrazione (giorni validi)': f'{percentuale_non_registrata:.1f}%'
        }
    else:
        valid_days = final_merged_df['date_label'].dropna().unique().tolist()

    detected_sessions = identify_nocturnal_sessions(final_merged_df)
    final_merged_df['night_id'] = None
    
    for idx, session in enumerate(detected_sessions, 1):
        night_label = f"Night_{idx}"
        mask = (final_merged_df['timestamp_iso'] >= session['start']) & (final_merged_df['timestamp_iso'] <= session['end'])
        final_merged_df.loc[mask, 'night_id'] = night_label

    metrics = calculate_all_metrics(final_merged_df, valid_days, detected_sessions)
    metrics.update(missing_metrics)
    
    missing_stats = {
        "minuti_non_registrati_tot": minuti_non_registrati_tot,
        "ore_non_registrate_tot": ore_non_registrate_tot,
        "percentuale_non_registrata": percentuale_non_registrata
    }    

    return {
        'num_dates': len(organized_files),
        'num_valid_days': len(valid_days),
        'num_valid_nights': len(detected_sessions),
        'final_merged_df': final_merged_df,
        'valid_days': valid_days,
        'detected_sessions': detected_sessions,
        'summary_metrics': metrics,
        'missing_stats': missing_stats
    }    

def identify_nocturnal_sessions(df):
    """Identify nocturnal sleep sessions and merge them if close enough"""
    if 'sleep-detection' not in df.columns:
        return []
    
    df_sorted = df.sort_values('timestamp_iso').reset_index(drop=True)
    sleep_col = 'sleep-detection'
    raw_sessions = []
    i = 0
    n = len(df_sorted)
    
    while i <= n - 20:
        window = df_sorted.iloc[i:i + 20]
        is_active = window[sleep_col].isin([101, 102])
        if is_active.sum() >= 15:
            start_idx = window[is_active].index[0]
            start_time = df_sorted.loc[start_idx, 'timestamp_iso']
            start_hour = start_time.hour
            if 6 <= start_hour < 19:
                i = start_idx + 1
                continue
            
            end_found = False
            for j in range(start_idx, n - 20):
                end_window = df_sorted.iloc[j:j + 20]
                is_zero = (end_window[sleep_col] == 0)
                if is_zero.sum() >= 15:
                    pre_window = df_sorted.iloc[start_idx:j]
                    pre_active = pre_window[sleep_col].isin([101, 102])
                    if not pre_active.empty and pre_active.any():
                        end_idx = pre_window[pre_active].index[-1]
                        end_time = df_sorted.loc[end_idx, 'timestamp_iso']
                        duration_hours = (end_time - start_time).total_seconds() / 3600

                        if duration_hours >= 2:
                            raw_sessions.append({'start': start_time, 'end': end_time})

                        i = j + 20
                        end_found = True
                        break
            
            if not end_found:
                break
        else:
            i += 1
            
    if not raw_sessions:
        return []

    merged_sessions = [raw_sessions[0]]
    MAX_AWAKE_HOURS_ALLOWED = 3.0 
    
    for current in raw_sessions[1:]:
        previous = merged_sessions[-1]
        time_between = (current['start'] - previous['end']).total_seconds() / 3600
        
        if time_between <= MAX_AWAKE_HOURS_ALLOWED:
            previous['end'] = current['end']
        else:
            merged_sessions.append(current)
            
    return merged_sessions                        

def calculate_all_metrics(df, valid_days, sessions):
    """Calculate all summary metrics"""
    metrics = {}
    metrics['Giorni Monitoraggio (Validati)'] = len(valid_days)
    metrics['Notti Monitoraggio (Validate)'] = len(sessions)
    
    # Steps analysis
    if 'step-counts' in df.columns:
        df_valid = df[df['date_label'].isin(valid_days)]
        steps_per_day = df_valid.groupby('date_label')['step-counts'].sum()
        if not steps_per_day.empty:
            metrics['Media Passi Giornalieri'] = f"{steps_per_day.mean():.2f}"
            metrics['% giorni > 7500 passi'] = f"{(steps_per_day > 7500).mean() * 100:.1f}%"
            metrics['% giorni > 10000 passi'] = f"{(steps_per_day > 10000).mean() * 100:.1f}%"
    
    # Activity intensity
    if 'activity-intensity' in df.columns:
        df_valid = df[df['date_label'].isin(valid_days)]
        intensity_counts = df_valid.groupby(['date_label', 'activity-intensity']).size().unstack(fill_value=0)
        for cat in ['LPA', 'MPA', 'VPA']:
            if cat not in intensity_counts.columns:
                intensity_counts[cat] = 0
        
        intensity_counts['MPA+VPA'] = intensity_counts['MPA'] + intensity_counts['VPA']
        num_valid = len(valid_days)
        if num_valid > 0:
            seven_day_mvpa = (intensity_counts['MPA+VPA'].sum() / num_valid) * 7
            metrics['7-day MVPA Projection (min)'] = f"{seven_day_mvpa:.2f}"
            metrics['Soglia 150 min (MVPA)'] = 'Raggiunta' if seven_day_mvpa >= 150 else 'Non Raggiunta'
    
    # Sedentary time
    if 'sleep-detection' in df.columns and 'activity-intensity' in df.columns:
        df_valid = df[df['date_label'].isin(valid_days)]
        df_awake = df_valid[~df_valid['sleep-detection'].isin([101, 102])]
        sedentary = df_awake[df_awake['activity-intensity'] == 'sedentary'].groupby('date_label').size()
        if not sedentary.empty:
            metrics['Sedentarietà giornaliera (ore)'] = f"{(sedentary.mean() / 60):.2f}"
            metrics['% Giorni Sedentari > 8h'] = f"{(sedentary >= 480).mean() * 100:.1f}%"
    
    # PAL
    if 'met' in df.columns:
        df_valid = df[df['date_label'].isin(valid_days)].dropna(subset=['met'])
        if not df_valid.empty:
            met_stats = df_valid.groupby('date_label')['met'].agg(['sum', 'count'])
            met_stats['PAL'] = met_stats['sum'] / met_stats['count']
            pal_mean = met_stats['PAL'].mean()
            
            if pal_mean < 1.40:
                categoria = "Inactive"
            elif 1.40 <= pal_mean <= 1.69:
                categoria = "Lightly active"
            elif 1.70 <= pal_mean <= 1.99:
                categoria = "Moderately active"
            else:
                categoria = "Highly active"
            
            metrics['PAL (Physical Activity Level)'] = f"{pal_mean:.2f}"
            metrics['Categoria PAL'] = categoria
    
    # MET-minutes
    if 'met' in df.columns:
        df_valid = df[df['date_label'].isin(valid_days)].copy()
        df_valid['met_filtrato'] = df_valid['met'].apply(lambda x: x if pd.notna(x) and x >= 3 else 0)
        met_sum = df_valid.groupby('date_label')['met_filtrato'].sum()
        total_met = met_sum.sum()
        num_valid = len(valid_days)
        
        if num_valid > 0:
            seven_day_met = (total_met / num_valid) * 7
            metrics['7-day MET-minutes Projection'] = f"{seven_day_met:.2f}"
            metrics['Soglia 500 MET-min'] = 'Raggiunta' if seven_day_met >= 500 else 'Non Raggiunta'
            metrics['Soglia 1000 MET-min'] = 'Raggiunta' if seven_day_met >= 1000 else 'Non Raggiunta'

    # Sleep analysis
    if 'sleep-detection' in df.columns and sessions:
        nightly_data = []
        for idx, session in enumerate(sessions, 1):
            night_label = f"Night_{idx}"
            night_subset = df[df['night_id'] == night_label]
            phase_counts = night_subset['sleep-detection'].value_counts().to_dict()
            nightly_data.append({
                'sonno': phase_counts.get(101, 0),
                'risveglio': phase_counts.get(102, 0)
            })
        
        if nightly_data:
            sleep_df = pd.DataFrame(nightly_data)
            avg_sleep = sleep_df['sonno'].mean()
            metrics['Durata Media Sonno (ore)'] = f"{(avg_sleep / 60):.2f}"
            perc_7_9 = ((sleep_df['sonno'] >= 420) & (sleep_df['sonno'] <= 540)).mean() * 100
            metrics['% di notti con 7-9 ore di sonno'] = f"{perc_7_9:.1f}%"
    
    # Physiological metrics during sleep
    if all(col in df.columns for col in ['pulse-rate', 'respiratory-rate', 'temperature', 'sleep-detection']):
        sleep_101 = df[(df['night_id'].notnull()) & (df['sleep-detection'] == 101)]
        if not sleep_101.empty:
            metrics['Freq. Cardiaca Media Notte (bpm)'] = f"{sleep_101['pulse-rate'].mean():.2f}"
            metrics['Freq. Respiratoria Media Notte (bpm)'] = f"{sleep_101['respiratory-rate'].mean():.2f}"
            metrics['Temperatura Media Notte (°C)'] = f"{sleep_101['temperature'].mean():.2f}"
    
    # HRV 24h
    if 'prv' in df.columns:
        df_valid = df[df['date_label'].isin(valid_days)].dropna(subset=['prv'])
        if not df_valid.empty:
            daily_prv = df_valid.groupby('date_label')['prv'].mean()
            metrics['24-h HRV'] = f"{daily_prv.mean():.2f}"
    
    # HRV nocturnal
    if 'prv' in df.columns and 'night_id' in df.columns:
        sleep_prv = df[(df['night_id'].notnull()) & (df['sleep-detection'] == 101)].dropna(subset=['prv'])
        if not sleep_prv.empty:
            nightly_prv = sleep_prv.groupby('night_id')['prv'].mean()
            metrics['Nocturnal HRV (ms)'] = f"{nightly_prv.mean():.2f}"
 
    # EDA Clean & Save back to master dataframe for plotting
    if 'eda' in df.columns:
        soglia_contatto = 0.01
        df['eda_pulito'] = df['eda']
        df.loc[df['eda_pulito'] < soglia_contatto, 'eda_pulito'] = np.nan
        
        if 'missing_value_reason' in df.columns:
            maschera_errore = df['missing_value_reason'].astype(str).str.contains('device_not_recording|error|detached', na=False, case=False)
            df.loc[maschera_errore, 'eda_pulito'] = np.nan

        df_valid = df[df['date_label'].isin(valid_days)]
        eda_awake = df_valid[df_valid['sleep-detection'] == 0]
        if not eda_awake['eda_pulito'].dropna().empty:
            metrics['Skin Conductance Level Veglia (microS)'] = f"{eda_awake['eda_pulito'].mean():.2f}"
        else:
            metrics['Skin Conductance Level Veglia (microS)'] = "Dato non disponibile"
        
        eda_sleep = df[(df['night_id'].notnull()) & (df['sleep-detection'].isin([101, 102]))]
        if not eda_sleep['eda_pulito'].dropna().empty:
            metrics['Skin Conductance Level Notte (microS)'] = f"{eda_sleep['eda_pulito'].mean():.2f}"
        else:
            metrics['Skin Conductance Level Notte (microS)'] = "Dato non disponibile"
            
    return metrics
   
def create_pdf_report(data, buffer):
    """Create PDF report with text on first page and graphs on second page in A4 format"""
    with PdfPages(buffer) as pdf:
        # Page 1: Text summary
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.95, 'Activity and Health Report', ha='center', fontsize=16, weight='bold')
        
        metrics = data['summary_metrics']
        y_position = 0.88
        
        for key, value in metrics.items():
            fig.text(0.1, y_position, f"{key}:", fontsize=10, weight='bold')
            fig.text(0.6, y_position, str(value), fontsize=10)
            y_position -= 0.03
            if y_position < 0.1:
                break
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Graphs
        df = data['final_merged_df']
        sessions = data['detected_sessions']
        
        plot_df = df.sort_values('timestamp_iso').copy()
        
        # Usiamo eda_pulito se disponibile per evitare crolli a zero artificiali
        target_eda_col = 'eda_pulito' if 'eda_pulito' in plot_df.columns else 'eda'
        cols_to_filter = [target_eda_col, 'pulse-rate', 'temperature', 'met']
        
        for col in cols_to_filter:
            if col in plot_df.columns:
                temp_series = plot_df[col].rolling(window=5, center=True).median()
                plot_df[col] = temp_series.ewm(span=10, adjust=False).mean()
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11.69, 8.27), sharex=True)
        axes = [ax1, ax2, ax3, ax4]
        
        def add_night_spans(ax_list, sessions):
            for ax in ax_list:
                for i, session in enumerate(sessions):
                    ax.axvspan(session['start'], session['end'], color='gray', alpha=0.2,
                              label='Nocturnal Session' if i == 0 else "")
        
        # EDA Plot con scala logaritmica (0 - 10 uS) e isolamento dei NaN
        if target_eda_col in plot_df.columns:
            ax1.plot(plot_df['timestamp_iso'], plot_df[target_eda_col], color='blue', linewidth=1)
            ax1.set_ylabel('EDA (uS)')
            ax1.set_title('Physiological and Metabolic Signals')
            ax1.grid(True, alpha=0.3, which='both')
            ax1.set_yscale('symlog', linthresh=0.01)
            ax1.set_ylim(0, 10) 
            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax1.set_yticks([0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        
        # Pulse Rate Plot
        if 'pulse-rate' in plot_df.columns:
            ax2.plot(plot_df['timestamp_iso'], plot_df['pulse-rate'], color='red', linewidth=1)
            ax2.set_ylabel('Pulse Rate (BPM)')
            ax2.set_ylim(30, 230)
            ax2.grid(True, alpha=0.3)
        
        # Temperature Plot
        if 'temperature' in plot_df.columns:
            ax3.plot(plot_df['timestamp_iso'], plot_df['temperature'], color='green', linewidth=1)
            ax3.set_ylabel('Temp (°C)')
            ax3.set_ylim(25, 45)
            ax3.grid(True, alpha=0.3)
        
        # MET Plot
        if 'met' in plot_df.columns:
            ax4.plot(plot_df['timestamp_iso'], plot_df['met'], color='purple', linewidth=1)
            ax4.set_ylabel('MET')
            ax4.set_xlabel('Time')
            ax4.set_yscale('symlog', linthresh=6)
            ax4.set_ylim(0, 20)
            ax4.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax4.set_yticks([0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20])
            ax4.grid(True, alpha=0.3)
        
        if sessions:
            add_night_spans(axes, sessions)
        
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right', fontsize='small')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

app = App(app_ui, server)
