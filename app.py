from datetime import datetime
from operator import index
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from streamlit_folium import st_folium
import folium
import geopy.distance   

APP_TITLE = 'Distance and Volume based Clustering'
APP_SUB_TITLE = 'This application helps in determining the first/last mile optimum centers basis centre of gravity model using euclidean distance'

def display_city_filter(df):
    city = df['Destination City'].drop_duplicates().sort_values()
    city_choice = st.sidebar.selectbox('Select your vehicle:', city)
    #st.header('')
    return city_choice

def display_cluster_selection():
    cluster_num = st.sidebar.slider('Select Num of Clusters',min_value=1,max_value=10)
    return cluster_num

def fileuploader():
    uploaded_file = st.sidebar.file_uploader(label='Choose your .csv file')
    return uploaded_file

def datadisplay(df):
    st.markdown('Below is the dataset for city choosen in the filter')
    datadisp = st.dataframe(df,use_container_width=True)
    return datadisp

def km(data,nclus):
    kmeans = KMeans(n_clusters=nclus,
                    random_state=0).fit(data.loc[data['Volume']>0, ['Latitude','Longitude']], 
                                                 sample_weight=data.loc[data['Volume']>0,'Volume'])
    cogs = kmeans.cluster_centers_
    cogs = pd.DataFrame(cogs, columns=['Latitude','Longitude'])
    data['Cluster'] = kmeans.predict(data[['Latitude', 'Longitude']])
    cogs = cogs.join(data.groupby('Cluster')['Volume'].sum())
    data = data.join(cogs, on='Cluster', rsuffix='_COG')
    st.markdown('Initial cluster Locations and total Volume served by those clusters')
    cogsdisp = st.dataframe(cogs,use_container_width=True)
    return cogs,cogsdisp,data

def color_opt():
    color_options = {'demand':'red',
                     'supply':'yellow',
                     'flow':'black',
                     'cog':'blue',
                     'candidate':'black',
                     'other':'gray'}
    return color_options

def initiate_map(data):
    m = folium.Map(location=data[['Latitude', 'Longitude']].mean(),
                   fit_bounds=[[data['Latitude'].min(),data['Longitude'].min()],
                               [data['Latitude'].max(),data['Longitude'].max()]])
    return m

def demand_map(data,map,color_options):
    # Add Volume points to the map
    for _, row in data.iterrows():
        folium.CircleMarker(location=[row['Latitude'],row['Longitude']],
                            radius=(row['Volume']**0.5),
                            color=color_options.get(str(row['Location Type']).lower(), 'gray'),
                            tooltip=str(row['Location Name'])+' '+str(row['Volume'])).add_to(map)    
    # Zoom based on volume points
    map.fit_bounds(data[['Latitude', 'Longitude']].values.tolist())
    return map

def clus_post(data,map,color_options):
    for _, row in data.iterrows():
            # New centers of gravity
            folium.CircleMarker(location=[row['Latitude'],
                                        row['Longitude']],
                                radius=(row['Volume']**0.5),
                                color=color_options['cog'],
                                tooltip=row['Volume']).add_to(map)
    return map

def flow_lines(data,map,color_options):
    for _, row in data.iterrows():
        # Flow lines
        if str(row['Location Type']).lower() in (['demand', 'supply']):
            folium.PolyLine([(row['Latitude'],
                            row['Longitude']),
                            (row['Latitude_COG'],
                            row['Longitude_COG'])],
                            color=color_options['flow'],
                            weight=(row['Volume']**0.5),
                            opacity=0.8).add_to(map)
    return map

def distance_from(loc1,loc2): 
    dis = (geopy.distance.geodesic(loc1, loc2).km)*1.35
    return round(dis,2)   

def total_vol(data):
    total = data['Volume'].sum()
    st.metric('Total Volume', total)
    
def seller_cnt(data):
    seller_cnt = data['Location Name'].count()
    st.metric('Seller Count',seller_cnt)

def pincode_cnt(data):
    pin_cnt = data['Destination_pincode'].nunique()
    st.metric('Pincode Count',pin_cnt)

def top_seller(data):
    top_sel = round(data['Volume'].max()/data['Volume'].sum()*100,2)
    top_sel_fin = top_sel.astype('str')+'%'
    st.metric('Top Seller Contribution',top_sel_fin)

def template_gen():
    template = pd.DataFrame(columns=['Location Name',
                                     'Location Type',
                                     'Destination City',
                                     'Destination_pincode',
                                     'Destination State',
                                     'Latitude',
                                     'Longitude',
                                     'Volume',
                                     'active_status',
                                     'curr_hub',
                                     'curr_lat',
                                     'curr_lon',
                                     'curr_destination'])
    return template

def download_data(data):
    return data.to_csv(index=False).encode('utf-8')


def main():
    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    #call function to generate template
    temp = template_gen()
    temp_fin = download_data(temp)

    #create template download button
    st.sidebar.download_button(
        label="Click Here To Download The Template",
        data=temp_fin,
        file_name='template.csv',
        mime='text/csv',
        )
    
    #call function to upload file and enable upload file button
    uploaded_file = fileuploader()
    
    if uploaded_file is not None:
        data_import = pd.read_csv(uploaded_file)
        
        city_choice = display_city_filter(data_import)

        filtered = data_import.loc[(data_import['Destination City'] == city_choice) & (data_import['Volume']>0)].sort_values(['Volume'],ascending=False)

        df_display = datadisplay(filtered)

        col1,col2,col3,col4 = st.columns(4)
        
        with col1:
            total_vol(filtered)
        with col2:
            seller_cnt(filtered)
        with col3:
            pincode_cnt(filtered)
        with col4:
            top_seller(filtered)

        clusnum = display_cluster_selection()

        color = color_opt()
        
        m = initiate_map(filtered)

        st.markdown('Below is the Demand Map of the chosen city. Please ensure all the demand points are within the same city. If not then please correct the data.')
        st.markdown('Bigger the circle, Higher the demand')
        
        init_map = demand_map(filtered,m,color)

        inital_map = st_folium(demand_map(filtered,m,color),width=1000)

        #Initialize kmeans cluster algorithm
        cogs,kmeans,filtered = km(filtered,clusnum)

        final_map1 = clus_post(cogs,init_map,color)
        final_map2 = st_folium(flow_lines(filtered,final_map1,color),width=1000)

            #setting up tuples for calculating distance
        filtered['origin'] = list(zip(filtered.Latitude,filtered.Longitude))
        filtered['destination'] = list(zip(filtered.Latitude_COG,filtered.Longitude_COG))

        #using distance_from function to calculated distance (a factor of 1.35 is included in the distance to offset for road distance)
        filtered['euc_dist']=filtered.apply(lambda row: distance_from(row.origin,row.destination),axis=1)
        #filtered["gg_dist"] = filtered.apply(get_gmaps_distance, axis=1)

        #Since a particular seller or customer can get split between multiple 
        clus_pos = filtered[['Cluster','destination']]
        clus_pos = clus_pos.drop_duplicates(['Cluster','destination'], keep='last')
        x =filtered.groupby(['Destination_pincode','Cluster'])['Volume'].sum()
        y = x.loc[x.groupby(level=0).idxmax()].reset_index()
        filtered = (pd.merge(filtered, y, on='Destination_pincode'))
        filtered = (pd.merge(filtered,clus_pos,left_on='Cluster_y', right_on='Cluster'))
        filtered['euc_dist_y']=filtered.apply(lambda row: distance_from(row.origin,row.destination_y),axis=1)
        filtered['euc_dist_curr']=filtered.apply(lambda row: distance_from(row.origin,row.curr_destination),axis=1)
        output = filtered[['Location Name','Destination City','Destination_pincode','Latitude','Longitude','Volume_x','Cluster_y','euc_dist_y']]

        output.rename({'Cluster_y': 'Clusters', 'euc_dist_y': 'stem_distance','Volume_x':'Volume'}, axis=1, inplace=True)

        mets = output.groupby(['Clusters']).agg({'Volume':'sum','stem_distance':'mean'}).reset_index()
        st.markdown('Final cluster locations and total volume served by those clusters with stem distance(i.e.:mean distance of each seller to cluster location)')

        mets1 = st.dataframe(mets,use_container_width=True)

        csv = download_data(output)

        curr_datetime = datetime.now()
        curr_datetime = curr_datetime.strftime('%m%d%Y')


        #create output download button
        st.sidebar.download_button(
            label="Click Here To Download The Output",
            data=csv,
            file_name='clustering_ouptut_'+city_choice+curr_datetime+'.csv',
            mime='text/csv',
            )

    else:
        'Please upload your file'
    
if __name__ == "__main__":
    main()




