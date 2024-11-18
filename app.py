import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.cluster import DBSCAN
from datetime import datetime
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    try:
        return pd.read_json('livedata.json')
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

def calculate_cluster_metrics(X, labels):
    metrics = {}
    
    non_noise_mask = labels != -1
    X_clean = X[non_noise_mask]
    labels_clean = labels[non_noise_mask]
    
    if len(np.unique(labels_clean)) >= 2:
        metrics['silhouette_score'] = silhouette_score(X_clean, labels_clean, metric='haversine')
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_clean, labels_clean)
        metrics['davies_bouldin_score'] = davies_bouldin_score(X_clean, labels_clean)
    
    metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
    metrics['n_noise_points'] = list(labels).count(-1)
    metrics['noise_ratio'] = metrics['n_noise_points'] / len(labels)
    
    return metrics

def get_infected_names_and_locations(input_name, df, epsilon=0.0158288, min_samples=2):
    X = df[['latitude', 'longitude']].values
    model = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine').fit(X)
    df['cluster'] = model.labels_
    
    metrics = calculate_cluster_metrics(X, model.labels_)
    
    input_name_clusters = []
    infected_data = {}
    
    for i in range(len(df)):
        if df['id'].iloc[i] == input_name:
            if df['cluster'].iloc[i] not in input_name_clusters:
                input_name_clusters.append(df['cluster'].iloc[i])
    
    for cluster in input_name_clusters:
        if cluster != -1:
            cluster_data = df[df['cluster'] == cluster]
            for _, row in cluster_data.iterrows():
                if row['id'] != input_name:
                    if row['id'] not in infected_data:
                        infected_data[row['id']] = {
                            'id': row['id'],
                            'latitude': row['latitude'],
                            'longitude': row['longitude'],
                            'cluster': cluster
                        }
    
    user_location = df[df['id'] == input_name][['latitude', 'longitude']].iloc[0]
    infected_list = list(infected_data.values())
    
    return infected_list, user_location, metrics

def visualize_contacts_and_clusters(input_name, infected_data, user_location, df):
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers',
        marker=dict(size=10, color=df['cluster'], colorscale='Viridis'),
        name='All Users',
        text=df['id'],
        hovertemplate="ID: %{text}<br>Cluster: %{marker.color}<extra></extra>"
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[user_location['latitude']],
        lon=[user_location['longitude']],
        mode='markers',
        marker=dict(size=15, color='blue', symbol='star'),
        name=f'{input_name} (You)',
        text=[input_name]
    ))

    if infected_data:
        infected_lats = [p['latitude'] for p in infected_data]
        infected_lons = [p['longitude'] for p in infected_data]
        infected_names = [p['id'] for p in infected_data]
        
        fig.add_trace(go.Scattermapbox(
            lat=infected_lats,
            lon=infected_lons,
            mode='markers',
            marker=dict(size=12, color='red'),
            name='Contacts',
            text=infected_names,
            hovertemplate="Contact: %{text}<extra></extra>"
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=user_location['latitude'], lon=user_location['longitude']),
            zoom=13
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        height=600
    )

    return fig

def add_user_to_dataset(new_name, lat, lon, df):
    current_timestamp = int(datetime.now().timestamp())
    new_entry = pd.DataFrame({
        'id': [new_name], 
        'latitude': [lat], 
        'longitude': [lon], 
        'timestamp': [current_timestamp]
    })
    df = pd.concat([df, new_entry], ignore_index=True)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce', downcast='integer')
    df = df.sort_values(by='timestamp', na_position='last').reset_index(drop=True)
    df.to_json('livedata.json', orient='records')
    return df

def main():
    st.set_page_config(page_title="Covid-19 Contact Tracking", layout="wide")
    
    df = load_data()
    
    img = Image.open('img.png')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(img, width=300)
    st.title("Covid-19 Contact Tracking App")

    st.sidebar.header("Advanced Settings")
    epsilon = st.sidebar.slider("Epsilon (km)", 0.5, 5.0, 1.6, 0.1) * 0.0099
    min_samples = st.sidebar.slider("Minimum Samples", 2, 3, 2)

    name = st.text_input("Enter Person Name")

    if name:
        if name not in df['id'].unique():
            st.error("Oops! This name does not exist in the dataset.")
            
            add_user = st.radio("Would you like to add yourself to the dataset?", ("No", "Yes"))
            
            if add_user == "Yes":
                col1, col2 = st.columns(2)
                with col1:
                    lat = st.number_input("Enter your latitude:", format="%.6f")
                with col2:
                    lon = st.number_input("Enter your longitude:", format="%.6f")
                
                if st.button("Add Me"):
                    if lat and lon:
                        df = add_user_to_dataset(name, lat, lon, df)
                        process_user_data(name, df, epsilon, min_samples)
                    else:
                        st.error("Latitude and longitude cannot be empty.")
        else:
            process_user_data(name, df, epsilon, min_samples)

def process_user_data(name, df, epsilon, min_samples):
    infected_data, user_location, metrics = get_infected_names_and_locations(name, df, epsilon, min_samples)

    tab1, tab2, tab3 = st.tabs(["Contact Results", "Clustering Metrics", "Analysis"])
    
    with tab1:
        if not infected_data:
            st.success("You have not been in contact with any affected person.")
        else:
            st.error("Oh no! You have been in contact with affected people.")
            st.write("Contacts found:")
            for idx, person in enumerate(sorted(infected_data, key=lambda x: x['id']), start=1):
                st.write(f"{idx}. {person['id']}")
        
        st.subheader("Contact Locations Map")
        fig = visualize_contacts_and_clusters(name, infected_data, user_location, df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Clustering Quality Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Clusters", metrics['n_clusters'])
        with col2:
            st.metric("Noise Points", metrics['n_noise_points'])
        with col3:
            st.metric("Noise Ratio", f"{metrics['noise_ratio']:.2%}")
        
        if 'silhouette_score' in metrics:
            st.write("Cluster Validation Scores:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
            with col2:
                st.metric("Calinski-Harabasz Score", f"{metrics['calinski_harabasz_score']:.1f}")
            with col3:
                st.metric("Davies-Bouldin Score", f"{metrics['davies_bouldin_score']:.3f}")
    
    with tab3:
        st.subheader("Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='longitude', y='latitude', color='cluster',
                           title='Spatial Distribution of Clusters')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            cluster_sizes = df[df['cluster'] != -1]['cluster'].value_counts()
            fig = px.bar(x=cluster_sizes.index, y=cluster_sizes.values,
                        title='Cluster Size Distribution',
                        labels={'x': 'Cluster', 'y': 'Number of Points'})
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()