import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import inject_theme, hero
from utils.model import get_default_experiment, load_dataset, class_name, get_model_probability, majority_vote

st.set_page_config(page_title="Prediction Studio | EnsembleLab",page_icon="🎯",layout="wide")
inject_theme()
hero("04 · TEST THE MODELS","Prediction Studio","Choose a held-out sample and see how each classifier predicts. Compare probability estimates and the ensemble vote.")
models,metrics,X_train,X_test,y_train,y_test=get_default_experiment()
st.caption("Default models · 80/20 split · 100 estimators · tree depth 3 · AdaBoost learning rate 0.5")
_,_,_,features=load_dataset()

mode=st.radio("Input mode",["Choose a test sample","Enter feature values"],horizontal=True)
if mode=="Choose a test sample":
    sample_id=st.selectbox("Select test sample",list(X_test.index),format_func=lambda x:f"Test record #{x}")
    row=X_test.loc[[sample_id]]
    actual=int(y_test.loc[sample_id])
    st.write("Actual label:", "🩷 Malignant" if actual==0 else "🩵 Benign")
else:
    st.info("Enter plausible numeric values for the 30 measurements. The defaults are the training-set medians.")
    medians=X_train.median()
    values={}
    with st.form("manual_prediction"):
        groups = {
            "Mean measurements": [f for f in features if f.startswith("mean ")],
            "Standard error measurements": [f for f in features if f.endswith(" error")],
            "Worst measurements": [f for f in features if f.startswith("worst ")],
        }
        for group, members in groups.items():
            with st.expander(group, expanded=group == "Mean measurements"):
                cols=st.columns(2)
                for i,feature in enumerate(members):
                    with cols[i%2]:
                        values[feature]=st.number_input(
                            feature, min_value=0.0, value=float(medians[feature]), format="%.4f",
                            help=f"Training range: {X_train[feature].min():.4f} to {X_train[feature].max():.4f}",
                        )
        submitted=st.form_submit_button("Run all four models",width="stretch")
    if not submitted:
        st.stop()
    row=pd.DataFrame([values],columns=features)
    outside = [f for f in features if values[f] < X_train[f].min() or values[f] > X_train[f].max()]
    if outside:
        st.warning("Outside the observed training range: " + ", ".join(outside) + ". Predictions may be less reliable.")
    actual=None

rows=[]
votes=[]
for name,model in models.items():
    pred=int(model.predict(row)[0])
    probs=get_model_probability(model,row)
    benign=probs.get(1,0.0)
    malignant=probs.get(0,0.0)
    votes.append(pred)
    rows.append({"Model":name,"Prediction":class_name(pred),
                 "Malignant probability":malignant,"Benign probability":benign,
                 "Confidence":max(malignant,benign)})
results=pd.DataFrame(rows)
st.markdown("#### Model-by-model predictions")
st.dataframe(results.style.format({
    "Malignant probability":"{:.1%}","Benign probability":"{:.1%}","Confidence":"{:.1%}"
}),width="stretch",hide_index=True)

benign_votes=votes.count(1)
malignant_votes=votes.count(0)
final=majority_vote(votes)
a,b,c=st.columns(3)
a.metric("Benign votes",f"{benign_votes}/4")
b.metric("Malignant votes",f"{malignant_votes}/4")
c.metric("Majority vote", "Tie" if final is None else class_name(final))
if final is None:
    st.info("2–2 tie: the models disagree. There is no majority prediction.")
chart=px.bar(results,x="Model",y=["Malignant probability","Benign probability"],barmode="group",
             color_discrete_sequence=["#ff6b8a","#36c5d8"])
chart.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=420,
                    yaxis_tickformat=".0%",margin=dict(l=15,r=15,t=35,b=15))
st.plotly_chart(chart,width="stretch")

if actual is not None:
    correct=sum(int(v==actual) for v in votes)
    st.metric("Models matching the actual test label",f"{correct}/4")
    if final is None:
        st.info("A tied vote cannot be scored as a majority prediction for this sample.")
    elif final==actual:
        st.success("The majority vote matches this test sample's label.")
    else:
        st.warning("The majority vote differs from this test sample's label.")
st.caption("Probabilities are model estimates, not guarantees. Majority voting here is a simple demonstration and is not a separately calibrated or clinically validated model.")
