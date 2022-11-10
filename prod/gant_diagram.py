# GANT DIAGRAM

import plotly.express as px

def draw_gant(df): #_sessions
    #df_sessions.head() #.drop(df_sessions.columns.difference(['start','stop','duration','description']), axis=1)

    fig = px.timeline(df, x_start="start", x_end="stop", y="subject")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        autosize=False,
        height=500,
        width=700, margin=dict(t=55, l=55, b=15, r=15)
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    # fig.show()
    return fig