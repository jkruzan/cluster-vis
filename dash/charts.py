from load_data import get_csv_df
import plotly.express as px
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.figure_factory as ff


def parallel_coordinates(features, df):
    # Cast cluster labels to int (required for parallel coordinates color)
    colored_feat = 'Cluster'
    df[colored_feat] = df[colored_feat].astype(int)

    # Get figure
    fig = px.parallel_coordinates(df, color=colored_feat,
                        dimensions=features)
    fig.update_layout(paper_bgcolor='rgb(0,0,0,0)')
    return fig

def scatter_matrix(features, df):
    # Get figure
    color = 'Cluster'
    df = df.copy()
    cluster_min = df['Cluster'].min()
    cluster_max = df['Cluster'].max()
    df['Cluster'] = 'Cluster ' + df['Cluster'].astype(str)
    cluster_order = {'Cluster':["Cluster "+ str(i) for i in range(cluster_min, cluster_max+1)]}
    # Get cluster oder changes df so make copy
    fig = px.scatter_matrix(df, color=color,
                            dimensions=features,
                            hover_data={df.index.name: df.index},
                            category_orders=cluster_order
                            )
    # Update traces (required for showing image on hover)
    fig.update_traces(hoverinfo='none',hovertemplate=None,)
    fig.update_layout(legend=dict(bgcolor='white'), paper_bgcolor='rgb(0,0,0,0)')
    return fig

def embed_scatter(df):
    '''
    Scatter plot for with embedded dimensions and color showing cluster label
    '''
    # (currently not sampling data)
    ##  Manually order the clusters for the legend
    # Find max and min of cluster labels
    cluster_min = df['Cluster'].min()
    cluster_max = df['Cluster'].max()
    # Convert integer labels to strings (required for discrete colors)
    df = df.copy()
    df['Cluster'] = 'Cluster ' + df['Cluster'].astype(str)
    cluster_order = {'Cluster':["Cluster "+ str(i) for i in range(cluster_min, cluster_max+1)]}

    # Get figure
    if 'Embed3' in df.columns:
        fig = px.scatter_3d(df, x='Embed1', y='Embed2', z='Embed3',
                        color='Cluster',
                        hover_data={df.index.name: df.index},
                        category_orders=cluster_order
                        )
    else:
        fig = px.scatter(df, x='Embed1', y='Embed2',
                        color='Cluster',
                        hover_data={df.index.name: df.index},
                        category_orders=cluster_order
                        )

    fig.update_traces(
        hoverinfo='none',
        hovertemplate=None,
    )
    fig.update_layout(autosize=False, margin=dict(t=10))
    return fig

def embed_scatter_heatmap(feature, df):
    '''
    Scatter plot for with embedded dimensions and color showing cluster label
    '''
    # (currently not sampling data)
    # Get figure
    if 'Embed3' in df.columns:
        fig = px.scatter_3d(df, x='Embed1', y='Embed2', z='Embed3',
                        color=feature)
    else:
        fig = px.scatter(df, x='Embed1', y='Embed2', 
                        color=feature)

    fig.update_traces(
        hoverinfo='none',
        hovertemplate=None,
    )
    fig.update_layout(autosize=False, margin=dict(t=10))
    fig.update_layout(width=700, height=500)
    return fig

def make_arrows(edge_x, edge_y, probs, edgetups):
    edge_x = np.array(edge_x).reshape((-1, 2))
    edge_y = np.array(edge_y).reshape((-1, 2))
    u = (edge_x[:,1]-edge_x[:,0])
    v = (edge_y[:,1]-edge_y[:,0])
    x = np.sum(edge_x, axis=1)*0.5
    y = np.sum(edge_y, axis=1)*0.5
    norm = np.sqrt(np.square(u)+np.square(v))
    un = u/norm
    vn = v/norm
    x1 = x+vn*0.05
    y1 = y-un*0.05
    x2 = x-vn*0.05
    y2 = y+un*0.05
    oldx = x
    oldy = y
    x = []
    y = []
    oldx1 = x1
    oldy1 = y1
    oldx2 = x2
    oldy2 = y2
    x1 = np.array([val for i in range(len(x1)) for val in [x1[i]-u[i]*0.5,x1[i]+u[i]*0.5,None] if probs[edgetups[i]] > 0.005])
    y1 = np.array([val for i in range(len(y1)) for val in [y1[i]-v[i]*0.5,y1[i]+v[i]*0.5,None] if probs[edgetups[i]] > 0.005])
    x2 = np.array([val for i in range(len(x2)) for val in [x2[i]-u[i]*0.5,x2[i]+u[i]*0.5,None] if probs[(edgetups[i][1], edgetups[i][0])] > 0.005])
    y2 = np.array([val for i in range(len(y2)) for val in [y2[i]-v[i]*0.5,y2[i]+v[i]*0.5,None] if probs[(edgetups[i][1], edgetups[i][0])] > 0.005])
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    colors = [i for i in range(int(len(x)/3))]
    un1 = np.array([un[i] for i in range(len(un)) if probs[edgetups[i]] > 0.005])
    vn1 = np.array([vn[i] for i in range(len(vn)) if probs[edgetups[i]] > 0.005])
    un2 = np.array([un[i] for i in range(len(un)) if probs[(edgetups[i][1], edgetups[i][0])] > 0.005])
    vn2 = np.array([vn[i] for i in range(len(vn)) if probs[(edgetups[i][1], edgetups[i][0])] > 0.005])
    #arrowx=[oldx[i] for i in range(len(oldx)) if probs[edgetups[i]] > 0.005]
    #y=[oldy[i] for i in range(len(oldy)) if probs[(edgetups[i][1], edgetups[i][0])] > 0.005]
    line_trace = go.Scatter(mode="lines", x=x, y=y,line=dict(color="rgb(255,0,0)"))
    fig = go.Figure(data=[line_trace])
    x12 = [oldx1[i] for i in range(len(oldx1)) if probs[edgetups[i]] > 0.005]
    y12 = [oldy1[i] for i in range(len(oldy1)) if probs[edgetups[i]] > 0.005]
    arrows1 = ff.create_quiver(x12-un1*(0.25-0.0625*0.5), y12-vn1*(0.25-0.0625*0.5), un1, vn1,line=dict(color="#ff0000"), scale=0.25,hovertext=None, hoverinfo=None)
    arrows2 = ff.create_quiver([oldx2[i] for i in range(len(oldx2)) if probs[(edgetups[i][1], edgetups[i][0])] > 0.005]+un2*(0.25-0.0625*0.5), [oldy2[i] for i in range(len(oldy2)) if probs[(edgetups[i][1], edgetups[i][0])] > 0.005]+vn2*(0.25-0.0625*0.5), -un2, -vn2,line=dict(color="#ff0000"), scale=0.25)
    annotations = []
    fig.add_traces(data=arrows1.data)
    fig.add_traces(data=arrows2.data)
    for i in range(len(oldx1)):
        if probs[edgetups[i]] > 0.005:
            annotations.append(dict(x=oldx1[i]+vn[i]*0.05, y=oldy1[i]-un[i]*0.05,
                                          xanchor='auto', yanchor='auto',
                                          text=f"{probs[edgetups[i]]:.2f}",
                                          font=dict(family='Arial',
                                                    size=12),
                                          showarrow=False))
    for i in range(len(oldx2)):
        if probs[(edgetups[i][1], edgetups[i][0])] > 0.005:
            annotations.append(dict(x=oldx2[i]-vn[i]*0.05, y=oldy2[i]+un[i]*0.05,
                                          xanchor='auto', yanchor='auto',
                                          text=f"{probs[(edgetups[i][1], edgetups[i][0])]:.2f}",
                                          font=dict(family='Arial',
                                                    size=12),
                                          showarrow=False))
    fig.update_layout(annotations=annotations)
    fig.update_layout(showlegend=False)
    return fig, annotations

def state_transition(df=None):
    print("LOADING DF FOR STATE TRANSITION")
    df = get_csv_df()
    clusters = list(set(df["Cluster"].values))
    clusters.sort()
    iddict = {}
    changes = {}
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if row["Cell ID"] not in iddict:
            iddict[row["Cell ID"]] = row["Cluster"]
        else:
            if iddict[row["Cell ID"]] not in changes:
                changes[iddict[row["Cell ID"]]] = [row["Cluster"]]
            else:
                changes[iddict[row["Cell ID"]]].append(row["Cluster"])
            iddict[row["Cell ID"]] = row["Cluster"]
    nclusters = len(clusters)
    G = nx.Graph()
    probs = {}
    for i in range(nclusters):
        G.add_node(int(clusters[i]))
    for i in range(nclusters):
        probs[(clusters[i], clusters[i])] = sum([1 if val==clusters[i] else 0 for val in changes[clusters[i]]])/len(changes[clusters[i]])
        for j in set(range(nclusters))-set([i]):
            probs[(clusters[i], clusters[j])] = sum([1 if val==clusters[j] else 0 for val in changes[clusters[i]]])/len(changes[clusters[i]])
            probs[(clusters[j], clusters[i])] = sum([1 if val==clusters[i] else 0 for val in changes[clusters[j]]])/len(changes[clusters[j]])
            G.add_edge(i, j)
    pos = nx.circular_layout(G)
    edge_x = []
    edge_y = []
    edgetups = []
    for edge in G.edges():
        edgetups.append((clusters[edge[0]], clusters[edge[1]]))
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        #edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        #edge_y.append(None)
    fig, annotations = make_arrows(edge_x, edge_y, probs, edgetups)
    """
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    """
    node_x = []
    node_y = []
    for node in G.nodes():
        #x, y = G.nodes[node]['pos']
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Reds',
            reversescale=True,
            color=[],
            size=60,
            colorbar=dict(
                thickness=15,
                title='Node',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    node_adjacencies = []
    node_text = []

    for node in G.nodes():
        node_adjacencies.append(node)
        node_text.append('Cluster Label '+str(clusters[node]))
        annotations.append(dict(x=pos[node][0], y=pos[node][1],
                                      xanchor='auto', yanchor='auto',
                                      text=f"{probs[(clusters[node], clusters[node])]:.2f}",
                                      font=dict(family='Arial',
                                                size=12),
                                      showarrow=False))
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    fig.add_trace(node_trace)
    fig.update_layout(width=800, height=800)
    fig.update_layout(annotations=annotations)
    return fig
def correlation_matrix(df):
    """
    Correlation matrix
    """
    ## TODO: Instead of using the given cluster, show all data
    # and order by cluster somehow. Maybe reindex using the cluster label? 

    # Get rid of unneeded features
    df = df.copy()
    df.drop(labels=['Cluster', 'Cell ID', 'Exp Cond', 'Time'], axis=1, inplace=True)
    # Map columns fom -2 to 2
    df = df.apply(lambda x: 2*np.tanh(x))
    # Sort in order of standard deviation
    std = df.std().sort_values(ascending=False)
    df = df.reindex(columns=std.index)

    #Plot it
    fig = px.imshow(df.transpose(),
                    x = df.index,
                    y = df.columns, 
                    color_continuous_scale='RdBu_r', 
                    origin='lower',
                    width=800, height=800)
    fig.update_layout(autosize=True)
    fig.update_xaxes(showticklabels=False,visible=False)
    return fig


def get_cluster_order(df):
    """
    Helper function to get the cluster labels in order from (min, max).
    Otherwise plotly would order by whatever appears first in the data
    """
    ##  Manually order the clusters for the legend
    # Find max and min of cluster labels
    df = df.copy()
    cluster_min = df['Cluster'].astype(int).min()
    cluster_max = df['Cluster'].astype(int).max()
    # Convert integer labels to strings (required for discrete colors)
    df['Cluster'] = 'Cluster ' + df['Cluster'].astype(str)
    return {'Cluster':[f'Cluster {i}' for i in range(cluster_min, cluster_max+1)]}
