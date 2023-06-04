import streamlit as st
import numpy as np
import plotly.express as px
from scipy.stats import norm
import torch

# Title
st.title("KL Divergence Demo")

x = np.linspace(-10, 10, 1000)
y = []

u = st.slider("Mean for P", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

s = st.slider("Standard Deviation for P", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

y.append(norm.pdf(x, u, s))

p = torch.distributions.Normal(u, s)

n = st.number_input("Number of Q distributions", min_value=1, max_value=10, value=1, step=1)

q = []

for i in range(n):
    u = st.slider(f"Mean for Q#{i+1}", min_value=-5.0, max_value=5.0, value=0.0, step=0.1, key="meanq" + str(i))

    s = st.slider(f"Standard Deviation for Q#{i+1}", min_value=0.1, max_value=5.0, value=1.0  , step=0.1, key="stdq" + str(i))

    q.append(torch.distributions.Normal(u, s))

for i in range(n):
    y.append(norm.pdf(x, q[i].mean.item(), q[i].stddev.item()))

y = np.array(y)

fig = px.line(title="KL Divergence")

for i in range(1, n+1):
    fig.add_scatter(x=x, y=y[i], name=f"Q#{i}")

fig.add_scatter(x=x, y=y[0], name="P")

st.plotly_chart(fig, use_container_width=True)

for i in range(n):
    st.write(f"KL Divergence(Q{i+1}||P): ", torch.distributions.kl_divergence(p, q[i]).item())
    st.write(f"KL Divergence(P||Q{i+1}): ", torch.distributions.kl_divergence(q[i], p).item())
    st.write("")