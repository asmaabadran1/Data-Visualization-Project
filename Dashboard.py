## Imports libraries
import plotly.express as ex
import numpy as np
from dash import Dash,html,dcc,Input, Output,State
#from jupyter_dash import JupyterDash
import pandas as pd
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.exceptions import PreventUpdate
import matplotlib.pyplot as plt
import plotly.io as pio
import base64
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
#tansorflow libraries
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO 
from PIL import Image
from dash import Dash,html,dcc,Input, Output,State
#from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc

np.random.seed(44)
pio.renderers.default = 'notebook'

# read data
img = "CatImagePrediction/image.jpg"
with open(img, 'rb') as f:
    img_data = f.read()
img_card_base64 = base64.b64encode(img_data).decode('utf-8')

breeds= ['Abyssinian',
 'Bengal',
 'Birman',
 'Bombay',
 'British',
 'Egyptian',
 'Maine',
 'Persian',
 'Ragdoll',
 'Russian',
 'Siamese',
 'Sphynx']

all_model_data=pd.read_csv("final_used_data/all_models_data.csv")
all_model_data["model_name"] = all_model_data["model_name"].apply(lambda x : str(x).split(".")[0]+"."+str(x).split(".")[1])
models = list(all_model_data["model_name"].unique())
model_path = "Models_h5"
# cats=pd.read_excel("catex.xlsx")
cats=pd.read_excel("final_used_data/size and hair.xlsx")
cats=cats.sort_values(by="Breed").reset_index()
cats_info=pd.read_csv("final_used_data/about_the_cat.csv")
cats_imgs=pd.read_csv("final_used_data/cat_images.csv")
numeric_columns=pd.read_excel("final_used_data/final_cat.xlsx")
numeric=numeric_columns.columns.tolist()[-12:]
cats[numeric]=numeric_columns[numeric]
cats.head()
cats=cats.drop("index",axis=1)
cats_info=cats_info.sort_values(by="Breed").reset_index()
cats_info=cats_info.drop("index",axis=1)


## Preprocess Data
tails_drop=cats_imgs.tail().index.values.tolist()
cats_imgs = cats_imgs.drop(index=[0, 1],axis=0)
cats_imgs=cats_imgs.drop(index=tails_drop,axis=0).reset_index()
cats_imgs.head()
# cats_imgs=cats_imgs.reset_index(drop=True)
cats_imgs=cats_imgs.drop("index",axis=1)
origin_to_continent = {'Abyssinia (Ethiopia)': 'Africa',
                       'Iran': 'Asia',
                       'Turkey': 'Asia and Europe',
                       'United Kingdom': 'Europe',
                       'United States': 'North America',
                       'Norway': 'Europe',
                       'Burma': 'Asia',
                       'Australia': 'Australia/Oceania',
                       'Greece': 'Europe',
                       'Russia': 'Europe and Asia',
                       'Brazil': 'South America',
                       'Canada': 'North America',
                       'Thailand': 'Asia',
                       'Egypt, South Asia': 'Africa and Asia',
                       'France, Syria': 'Europe and Asia',
                       'United States, United Kingdom': 'North America and Europe',
                       'Singapore': 'Asia',
                       'Japan': 'Asia',
                       'Isle of Man': 'Europe',
                       'Burma / Myanmar': 'Asia',
                       'Germany': 'Europe',
                       'United States, Africa': 'North America and Africa',
                       'Sweden': 'Europe',
                       'Egypt': 'Africa',
                       'Ukraine': 'Europe',
                       'China': 'Asia',
                       'Kenya': 'Africa'}


# replace values using the dictionary
cats['Continents'] = cats['Origin'].replace(origin_to_continent)

# print the updated DataFrame
cats.info()
cats["Continents"]=cats["Continents"].convert_dtypes(str)
cats["Origin"]=cats["Origin"].convert_dtypes(str)
cats["Breed"]=cats["Breed"].convert_dtypes(str)
cats=cats.drop("Hair",axis=1)
breeds=[]
cats_imgs_sorce=[]
for i in range(cats.shape[0]):
    breed=cats["Breed"][i].split("Cat")[0]
    breeds.append(breed)
    cats_imgs_sorce.append(cats_imgs.loc[i].tolist()[0])
nums=cats.columns.tolist()[-5:-1]
#create new df for map
map_df=pd.DataFrame([cats["ISO-ALPHA 3"].values,cats["Continents"].values,cats["Origin"].values,cats_imgs_sorce,breeds],index=["ISO_ALPHA 3","Continents","Origin","img","breed"]).T
map_df["Description"]=cats_info["Description"]
map_df.info()

map_df[nums]=cats[nums]
cats.info()

map_df["Description"]=cats_info["Description"]
cats["Breed"]=cats["Breed"].sort_values()
bar_df=pd.DataFrame()
bar_df[["Breed","Size","Origin","Life Expectancy","Weight","Height","Body Length"]]=cats[["Breed","Size","Origin","avr_Life Expectancy","avr_Weight","avr_Height","avr_Body Length"]]

# create figure
# treemap=ex.treemap(map_df,path=["Continents","Origin","breed"],color_continuous_scale='RdBu',color="Origin",hover_name="breed", title="Cat",custom_data=["img","Description"],  color_discrete_map={'Continents':'lightgrey', 'United States':'gold', 'Dinner':'darkblue'})
treemap=ex.treemap(map_df,path=["Continents","Origin","breed"],color="Origin",hover_name="breed", title="Cat",custom_data=["img","Description"], color_discrete_sequence=["#F5651F","#EBAB10","#EBD7A7","beige"] ,color_continuous_scale=ex.colors.sequential.Oranges)

treemap=treemap.update_layout(
    height=500,
    title={
        'text': " ",
        'y':1,
        'x':0.10,
        'xanchor': 'center',
        'yanchor': 'top',
#         'pad':dict(l=20, r=20, t=20, b=0)
        }, 
        font=dict(
        family="Droid Sans Mono, sans-serif",
        color="RebeccaPurple",
        size=16,
       ),

            margin=dict(l=20, r=20, t=20, b=20),
        clickmode='event+select',\


)


#fix the color schema
bar_df.sort_values("Weight",ascending=True).tail(20)
pio.templates

bar=ex.bar(bar_df.sort_values(by="Weight",ascending=False).head(20),x="Weight",y="Breed",title="Breed by {}",color="Weight",color_continuous_scale=ex.colors.sequential.Oranges)


figu=ex.bar(bar_df.sort_values(by="Life Expectancy",ascending=True).head(20),x="Life Expectancy",y="Breed",title=f"Breed by",color="Life Expectancy",color_continuous_scale=ex.colors.sequential.Oranges)
ig=ex.bar(bar_df.sort_values(by="Life Expectancy",ascending=False).tail(20),x="Life Expectancy",y="Breed",title=f"Breed by ",color="Life Expectancy",color_continuous_scale=ex.colors.sequential.Oranges)


# Main code for dashboard 😎
app = Dash(__name__,external_stylesheets=[dbc.themes.LUMEN],suppress_callback_exceptions=True) #suppress_callback_exceptions=True
server = app.server
app.layout=html.Div(children=[
    
            html.H1(html.B("😸Purrfect Analytics:A Dashboard for Exploring the World of Cats"), style={'color': 'black', 'fontSize': 25, 'textAlign':'center','font-family':'Courier New, monospace','margin':'20px'}),

            #1.Contorls
            html.Div([
                #change the theme of the button theme 
                html.H3("Insights",style={'color': 'black', 'fontSize': 24,'textAlign':'center','font-family':'Courier New, monospace','display':'inline-block'}),
                daq.ToggleSwitch(id='my-toggle-switch',value=False,color="Orange",style={'display':'inline-block','margin-right':'20px','margin-left':'40px'}),
                html.H3("Predict The Breed",style={'color': 'black', 'fontSize': 24,'textAlign':'center','font-family':'Courier New, monospace','display':'inline-block'}),
            ],style={"textAlign":"center","padding-left":"80px"}),

            html.Hr(style={'width':'10','textAlign':'center'}),

    
 #---------------------------------------------------------------------------------------------------------------   
    
        #Body
          html.Div([ 
              #the function switch_dashboard will send the code here for the two modes we have



          ],style={"padding-left":"20px","padding-right":"20px"},id="body")
          
      ])


#define the callback function for the bar and the dropdown menu and the descending and ascending
@app.callback(
    Output(component_id="body",component_property="children"),
    Input(component_id="my-toggle-switch",component_property="value")
    #when toggle value becomes true switch to predict dashboard
)
def switch_dashboard(toggle_state):
  #insights dashboard components
        info_card = dbc.Card([
                dbc.Row([
                           dbc.Col(
                                      html.Img(src="https://cats.com/wp-content/uploads/2022/08/cat-love-bites-compressed.jpg",className="img-fluid rounded-start",style={
                                      "borderTopLeftRadius": "20px",
                                      "borderBottomLeftRadius": "20px",
                                      "borderTopRightRadius": "0px",
                                      "borderBottomRightRadius": "0px",
                                      "width": "100%",
                                      "height": "100%",
                                      "objectFit": "cover"
                                                }),width=3
                                  ),
                            dbc.Col(
                                      [
                                      dbc.CardBody([
                                          html.H4(html.B("Cats are as anicent as History!"), className="card-title",style={'fontSize': 25, 'font-family':'Courier New, monospace'}),
                                          html.P("Cats are as ancient as history itself. They have been a part of human civilization for thousands of years, with evidence of domesticated cats dating back to ancient Egypt. Cats were revered in ancient Egypt, where they were worshiped as sacred animals and often depicted in art and literature. They were also used to control pests in households and on farms. Today, cats remain one of the most popular pets worldwide, with millions of households around the world owning at least one feline friend.", className="card-text")
                                      ]
                                      )
                        ],width=8)
                      ],)
                ],style={"background-color":"#EBD7A7","border":"0px","margin-bottom":"20px",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)'},outline=False)  

        #insights card
        insights_card = dbc.Card([
                              dbc.Row([
                                          dbc.Col(html.P( html.B('Cat breeds insights'),style={'padding-right':'10px','color': 'black', 'fontSize': 25, 'font-family':'Courier New, monospace',"margin-top":"10px"} ),width={"size":"auto","offset":"1"},style={"align-items":"center"}),
                                          dbc.Col([dcc.Dropdown(
                            id='dropdown',
                            options=[{'label': i, 'value': i} for i in [ 'Life Expectancy', 'Weight', 'Height',
                                                                                           'Body Length']],
                            value='Weight',style={'padding-left':'10px','padding-right':'10px',"margin-top":"10px"}
                        )],width=3,style={"align-items":"center"}),
                                                          dbc.Col(dcc.Checklist(
                                                                                    id='checkbox',
                                                                                    options=[
                                                                                        {'label': 'Ascending', 'value': 'checkbox_value'}
                                                                                    ],
                                                                                    value=[]
                                                                                ,style={"margin-top":"15px"}),width=3,style={"align-items":"center"} )
                                                      ]),
                                              dbc.Row(
                                                          dbc.CardBody([dcc.Graph(id="bar")],style={"margin-top":"20px"})
                                                    )
                                          ],outline=False,style={"border":"0px",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)'},)
        
        #tree_map
        tree_map=dbc.Row([
             dbc.Col( [dbc.Card([dcc.Graph(id="map_plot",figure=treemap)],body=True,
                                style={"align":"center","width":"100%","height":"100%",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)',})],width=6),\
             dbc.Col([dbc.Card([
                    html.Img(src=f'data:image/jpg;base64,{img_card_base64}',
                    style={"object-fit": "cover","width":"340", "height":"160","border-radius":"20px","margin-left":"20px","margin-top":"20px"}),
                    html.H4(html.B("click on the tree to check different breeds",style={'font-family':'Courier New, monospace'}))],
                    style={"align-items":"center","border":"0px","background-color":"#EBD7A7",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)'}),]
                    ,id='cat_img',style={"align-items":"center"},width=6)],justify="center")
        #map callback function


      
    #--------------------------------------------------------------------------------------------------------------------
        
        predict_part =html.Div(children=[
    # tittle card
    dbc.Card([html.H3(html.B("Try InceptionV3 pretrained model"),
                      style={"margin":"20px",'textAlign':'center','font-family':'Courier New, monospace'})]
                      ,id="predict_card",style={"margin":"20px",'border':'0px','background-color':'#EBD7A7','box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)',} ),    
    #choose hyperparamters
    dbc.Card([
        # tittle row 
        dbc.Row([
        dbc.Col( html.H3(html.B("choose hyperparamter"), 
                 className="card-title",
                 style={'fontSize': 25, 'font-family':'Courier New, monospace',
                 'margin-top':'10px','margin-bottom':'10px','textAlign':'center'})
                ,align="center")],justify="center",align="center",style={"align-items":"center"}),
        dbc.Row([
                dbc.Col([dbc.Card([html.H5(html.B("Optimizer"),style={"margin":"7px",'font-family':'Courier New, monospace'}),
                dcc.Dropdown(["RMSprop","Adam","SGD"],value=["Adam"],style={"margin":"7px","margin-right":"20px"},id="optimizer",multi=True)],
                style={"margin":"7px",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)',})],width={"size":"5"}, id="optimizer",style={"margin":"10px"}),
                dbc.Col([dbc.Card([html.H5(html.B("learning rate"),style={"margin":"7px",'font-family':'Courier New, monospace'}),dcc.Dropdown([0.001,0.01,0.1],value=[0.1],id="learning_rate",style={"margin":"7px","margin-right":"20px"},multi=True)],style={"margin":"7px",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)'})],width={"size":"5"},style={"margin":"10px"}),
                
              ],align="center",style={"align-items":"center"},justify="center"),
       dbc.Row([dbc.Col(html.Button(html.B("Train"),id="train_btn",className="btn btn btn-warning",style={"margin-bottom":"10px","margin-top":"10px","width":"400px"}),
       width=4,align="center")],justify="center"),
       
        dbc.Row([
                   
                dbc.Col([dcc.Graph(id="accuracy_graph")],width=6),
                dbc.Col([dcc.Graph(id="loss_graph")],width=6)
                ],justify="center"),

    ],style={"margin":"20px",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)'}),
    # model prediction
    dbc.Card([
    dbc.Row([html.H3(html.B("Make prediction"),style={"margin":"10px",'textAlign':'center','border':'20px'})],id="predict_card",style={"margin":"20px",'font-family':'Courier New, monospace'} ),dbc.Row([
        dbc.Col([   #first half upload picture
                    dbc.Card([html.H3(html.B("upload Image"),
                    style={'fontSize': 25, 'font-family':'Courier New, monospace','margin-top':'10px','margin-bottom':'10px','textAlign':'center'}),
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                                        ]),
                                            style={
                                                'width': '90%',
                                                'height': '50px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin':'7px'
                                                },
                                                # Allow multiple files to be uploaded
                                                #multiple=True
                                            )],style={"margin-left":"20px","margin-top":"10px",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)'}),
                html.Div(id="uploaded_img"),
                dbc.Row([dbc.Col(html.Button(html.B("predict"),id="test_btn",className="btn btn btn-warning",style={"margin-bottom":"10px","margin-top":"10px","width":"400px"}),
                        width=8,align="center")],justify="center"),
                ],width=6),
            
        
                dbc.Col([
                        dbc.Card([html.H4(html.B("choose model"),style={"margin":"10px",'textAlign':'center','font-family':'Courier New, monospace'}),
                        dcc.Dropdown(models,value=models[0],style={"margin":"7px","margin-right":"20px"},id="model_choice")],style={"margin":"10px",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)',}),
                        dcc.Graph(id='prediction_graph'),
                        ],width=6)

                        ]) 
                ],style={"margin-right":"20px",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)'},),  

        
])
          

       



        #predict the breed dashboard
        if toggle_state ==True: #first card ↓                               
             return  dbc.Row(predict_part)
          
             








        if toggle_state ==False:
            return dbc.Row(dbc.Col(info_card,width=12),justify="center"),dbc.Row([dbc.Row(html.H3("Cats Breed origin",style={'color': 'black', 'fontSize': 24,'textAlign':'center','font-family':'Courier New, monospace'})), 
             dbc.Row( dbc.Col(tree_map,width=12),justify="center"),  
           ],style={"margin-top":"40px","margin-bottom":"40px"}) ,  dbc.Row([
            dbc.Col(insights_card, width=9)
        ], justify="center")
    #  html.Div([dcc.Graph(figure=ig)],

#----------------------------------------------------------Callouts---------------------------------------------------------    
#bar callout
@app.callback(
Output('bar','figure'),
Input('dropdown','value'),
Input('checkbox','value'))
def update_bar_graph(d_value,c_value):
    #if false head
    #if true tail
    if c_value:
        fig=ex.bar(bar_df.sort_values(by=d_value,ascending=False).tail(20),x=d_value,y="Breed",title=f"Breed by {d_value}",color=d_value,color_continuous_scale=ex.colors.sequential.Oranges)
        if d_value == "Life Expectancy":
            fig.update_traces(hovertemplate="%{x:} years <br>")
        if d_value == "Weight":
            fig.update_traces(hovertemplate="%{x:} pounds <br>")
        
        if d_value == "Height":
            fig.update_traces(hovertemplate="%{x:} inches <br>")
        if d_value == "Body Length":
            fig.update_traces(hovertemplate="%{x:} inches <br>")
        


        return fig
    else :
        fig= ex.bar(bar_df.sort_values(by=d_value,ascending=True).head(20),x=d_value,y="Breed",title=f"Breed by {d_value}",color=d_value,color_continuous_scale=ex.colors.sequential.Oranges)
        if d_value == "Life Expectancy":
            fig.update_traces(hovertemplate="%{x:} years <br>")
        if d_value == "Weight":
            fig.update_traces(hovertemplate="%{x:} pounds <br>")
        
        if d_value == "Height":
            fig.update_traces(hovertemplate="%{x:} inches <br>")
        if d_value == "Body Length":
            fig.update_traces(hovertemplate="%{x:} inches <br>")
        return fig
        
        
        
        

def display_magimg(clickData):



        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        # Define image URL
        if "customdata" not in clickData["points"][0]:
            return ''
        else:
            i=clickData["points"][0]["customdata"][0]
            print(clickData["points"][0]["hovertext"])
            if(clickData["points"][0]["hovertext"]) =='(?)':
                return ''
            # Request image data
            else:
                with open(i, 'rb') as f:
                        img_data = f.read()
                # Convert image data to base64
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                cat=dbc.Card([
                    html.Img(src=f'data:image/jpg;base64,{img_base64}',style={"object-fit": "cover","width":"340", "height":"160","border-radius":"20px","margin-left":"20px","margin-top":"20px"}),html.H3(html.B(clickData["points"][0]["hovertext"],style={'font-family':'Courier New, monospace'})),dbc.CardBody(html.P(clickData["points"][0]["customdata"][1]))],
                    style={"align-items":"center","border":"0px","background-color":"#EBD7A7",'box-shadow': '0 2px 4px 0 rgba(0,0,0,0.2)'}),
                # Print base64 string
                return cat



@app.callback(
Output('cat_img', 'children'),
Input('map_plot', 'clickData'))
def update_output(clickData):
    if clickData:
           return display_magimg(clickData)
    else:
        raise PreventUpdate

        

def parse_contents(contents, filename, date):
    return html.Div([


        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents,width="340", height="160",style={"object-fit": "cover","margin-top":"20px"}),


    ])


##____________________________________________Prediction Callbacks_______________________________________
def show_fig(df, type1):
    fig = px.line(df, x="epoch", y=type1, color="model_name",template="ggplot2") 
    fig.update_layout(
        height=600,
        width=600,
        hovermode='x unified',
        updatemenus=[
            dict(
                type = "buttons",
                direction = "up",
                
            )
        ],
        margin=dict(l=5, r=10, t=5, b=5),
        legend=dict(x=0.5, y=-0.1, orientation='h')
    )
    return fig

def make_prediction(value):
    breeds= ['Abyssinian','Bengal','Birman','Bombay','British','Egyptian','Maine','Persian','Ragdoll','Russian','Siamese','Sphynx']
    with open('data.txt', encoding='UTF8') as f:
        contents = f.read()
    decoded_data=base64.b64decode(str(contents).split(',')[1])

    # load the saved model
    model = tf.keras.models.load_model(os.path.join(model_path , value + ".h5"))

    # load and preprocess the image
    img = Image.open(BytesIO(decoded_data))
    img = img.resize((299, 299)) # resize the image to match the input size of the model
    img_array = np.array(img) / 255.0 # convert the image to a NumPy array and normalize the pixel values

    # make a prediction
    y_pred = model.predict(np.expand_dims(img_array, axis=0))
    fig = ex.bar(x=list(y_pred.reshape(12,-1).flatten()),color=list(y_pred.reshape(12,-1).flatten()), y=breeds,orientation='h',color_continuous_scale=ex.colors.sequential.Oranges)
    fig.update_layout(
        xaxis_title='',
        yaxis_title='',
        hovermode='x unified',
        updatemenus=[
            dict(
                type = "buttons",
                direction = "left",
                
            )
        ],
        margin=dict(l=5, r=30, t=5, b=5)
    )
    return fig

def parse_contents(contents, filename):
    with open('data.txt', 'w') as f:
        f.write(contents)
    return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
       dbc.Card([
                    html.Img(src=contents,style={"object-fit": "cover","max-width": "90%","max-height": "90%","width":"340", "height":"160","border-radius":"20px","margin-left":"20px","margin-top":"20px","margin-right":"20px"}),
                    html.H3(html.B(str(filename).split(".")[0],style={'font-family':'Courier New, monospace'}))],
                    style={"align-items":"center","border":"0px","background-color":"#EBD7A7","margin":"20px"})
                    
  
        ])

@app.callback(
     Output("accuracy_graph","figure"),
     Output("loss_graph","figure"),
     Input("train_btn","n_clicks"),
     State('learning_rate', 'value'),
     State('optimizer','value'),
     )
def display_value(n_clicks, value1, value2):
    sub_data = all_model_data.loc[((all_model_data["learning_rate"].isin(value1)) & (all_model_data["optimizer"].isin(value2)))]
    fig1 = show_fig(sub_data, "accuracy")
    fig2 = show_fig(sub_data, "loss")
    return fig1, fig2
    
@app.callback(Output('uploaded_img', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output_image(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = parse_contents(list_of_contents, list_of_names)
        return children 
    return ""

@app.callback(
              Output("prediction_graph",'figure'),
              Input("test_btn","n_clicks"),
              State("model_choice","value"))
def update_output_image(n_clicks,value):
    fig = make_prediction(value) 
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

#  thank you ✨🤍
