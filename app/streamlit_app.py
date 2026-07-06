import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Student Career Path Predictor",
    page_icon="",
    layout="centered",
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_artifacts():
    return {
        'model':         joblib.load(os.path.join(MODEL_DIR,'best_ml_model.pkl')),
        'label_encoders':joblib.load(os.path.join(MODEL_DIR,'label_encoders.pkl')),
        'target_encoder':joblib.load(os.path.join(MODEL_DIR,'target_encoder.pkl')),
        'scaler':        joblib.load(os.path.join(MODEL_DIR,'scaler.pkl')),
        'feature_cols':  joblib.load(os.path.join(MODEL_DIR,'feature_cols.pkl')),
        'ordinal_map':   joblib.load(os.path.join(MODEL_DIR,'ordinal_map.pkl')),
        'binary_cols':   joblib.load(os.path.join(MODEL_DIR,'binary_cols.pkl')),
        'nominal_cols':  joblib.load(os.path.join(MODEL_DIR,'nominal_cols.pkl')),
    }

artifacts = load_artifacts()

RECS = {
    "Applications Developer": {
        "icon":"💻","salary":"₹4–12 LPA","growth":"High",
        "description":"Build desktop and enterprise applications used by businesses worldwide.",
        "courses":["Android Development with Kotlin — Udacity","iOS App Development with Swift — Coursera","Full Stack Web Development — freeCodeCamp","Java Programming Masterclass — Udemy"],
        "skills":["Java / Kotlin","Swift","REST APIs","Git","OOP"],
        "certs":["Google Associate Android Developer","AWS Certified Developer","Oracle Java SE"],
    },
    "CRM Technical Developer": {
        "icon":"🤝","salary":"₹5–14 LPA","growth":"High",
        "description":"Develop and customise CRM platforms to manage customer relationships at scale.",
        "courses":["Salesforce Platform Developer — Trailhead (Free)","CRM Fundamentals — Coursera","SQL for Data Analysis — Mode Analytics","HubSpot CRM Certification — HubSpot Academy (Free)"],
        "skills":["Salesforce","SQL","Apex","Data Modelling","APIs"],
        "certs":["Salesforce Certified Platform Developer I","HubSpot CRM Certification"],
    },
    "Database Developer": {
        "icon":"🗄️","salary":"₹4–13 LPA","growth":"Moderate",
        "description":"Design, build, and optimise databases that power modern applications.",
        "courses":["SQL and Database Design — Coursera","MongoDB for Developers — MongoDB University (Free)","PostgreSQL for Everybody — Coursera","Oracle Database Fundamentals — Oracle Academy"],
        "skills":["SQL","NoSQL","Database Design","Query Optimisation","ETL"],
        "certs":["Oracle Database SQL Certified Associate","MongoDB Certified Developer"],
    },
    "Mobile Applications Developer": {
        "icon":"📱","salary":"₹5–16 LPA","growth":"Very High",
        "description":"Create apps for iOS and Android platforms used daily by billions.",
        "courses":["Flutter & Dart — The Complete Guide (Udemy)","React Native — Zero to Mastery","Android Jetpack Masterclass — Udemy","iOS & Swift Bootcamp — Udemy"],
        "skills":["Flutter","Dart","Swift","Kotlin","UI/UX Basics"],
        "certs":["Google Associate Android Developer","Apple iOS Developer Certificate"],
    },
    "Network Security Engineer": {
        "icon":"🔒","salary":"₹6–20 LPA","growth":"Very High",
        "description":"Protect networks and systems from cyber threats and security breaches.",
        "courses":["Google Cybersecurity Certificate — Coursera","CompTIA Security+ — Professor Messer (Free)","Ethical Hacking — Udemy","Network Security — Cisco NetAcad (Free)"],
        "skills":["Firewalls","Penetration Testing","Linux","TCP/IP","SIEM"],
        "certs":["CompTIA Security+","Certified Ethical Hacker (CEH)","CISSP"],
    },
    "Software Developer": {
        "icon":"⚙️","salary":"₹4–15 LPA","growth":"High",
        "description":"Design and write software that solves real-world problems across industries.",
        "courses":["CS50 — Harvard (Free)","Python for Everybody — Coursera","Data Structures & Algorithms — NeetCode (Free)","Clean Code Principles — Udemy"],
        "skills":["Python / Java / C++","OOP","Algorithms","Git","Debugging"],
        "certs":["Oracle Java SE Certification","Microsoft Certified: Azure Developer"],
    },
    "Software Engineer": {
        "icon":"🏗️","salary":"₹6–25 LPA","growth":"Very High",
        "description":"Architect and build scalable, reliable software systems from the ground up.",
        "courses":["System Design — Grokking (Educative)","OS — Neso Academy (Free)","Agile and Scrum — Udemy","Software Engineering Fundamentals — edX"],
        "skills":["System Design","Agile/Scrum","CI/CD","Cloud Basics","Microservices"],
        "certs":["AWS Certified Solutions Architect","Google Professional Cloud Developer"],
    },
    "Software Quality Assurance (QA) / Testing": {
        "icon":"✅","salary":"₹3–12 LPA","growth":"Moderate",
        "description":"Ensure software quality by designing and executing comprehensive test strategies.",
        "courses":["Software Testing Fundamentals — Coursera","Selenium with Python — Udemy","API Testing with Postman — Udemy","ISTQB Foundation Prep — Udemy"],
        "skills":["Manual Testing","Selenium","Test Planning","API Testing","JIRA"],
        "certs":["ISTQB Certified Tester Foundation Level","Certified Agile Tester"],
    },
    "Systems Security Administrator": {
        "icon":"🛡️","salary":"₹5–18 LPA","growth":"High",
        "description":"Administer and secure an organisation's systems, servers, and infrastructure.",
        "courses":["CompTIA Security+ — Professor Messer (Free)","Linux Admin Bootcamp — Udemy","Active Directory — Microsoft Learn (Free)","CompTIA CySA+ — Udemy"],
        "skills":["Linux","Active Directory","Incident Response","Firewalls","VPN"],
        "certs":["CompTIA Security+","CompTIA CySA+","CISA"],
    },
    "Technical Support": {
        "icon":"🎧","salary":"₹2.5–8 LPA","growth":"Moderate",
        "description":"Help users resolve technical issues and keep systems running smoothly.",
        "courses":["Google IT Support Certificate — Coursera","CompTIA A+ — Professor Messer (Free)","ITIL Foundation — Axelos","Help Desk Support — LinkedIn Learning"],
        "skills":["Troubleshooting","Networking Basics","Customer Communication","Ticketing"],
        "certs":["CompTIA A+","Google IT Support Certificate","ITIL Foundation"],
    },
    "UX Designer": {
        "icon":"🎨","salary":"₹4–18 LPA","growth":"High",
        "description":"Design intuitive and beautiful digital experiences that users love.",
        "courses":["Google UX Design Certificate — Coursera","Figma UI/UX Essentials — Udemy","Human-Computer Interaction — Coursera","UX Research — Interaction Design Foundation"],
        "skills":["Figma","Wireframing","User Research","Prototyping","Design Thinking"],
        "certs":["Google UX Design Certificate","Nielsen Norman Group UX Certification"],
    },
    "Web Developer": {
        "icon":"🌐","salary":"₹3–15 LPA","growth":"High",
        "description":"Build websites and web applications that power the modern internet.",
        "courses":["Web Developer Bootcamp — Udemy (Colt Steele)","Full Stack Open — Univ. of Helsinki (Free)","JavaScript — freeCodeCamp (Free)","Node.js & Express — Udemy"],
        "skills":["HTML/CSS","JavaScript","React","Node.js","Databases"],
        "certs":["Meta Front-End Developer Certificate","AWS Certified Developer"],
    },
}

st.title("Student Career Path Predictor")
st.write("Fill in your details below and click **Predict** to find your best-fit career.")
st.divider()

tab1, tab2 = st.tabs(["Predict My Career", "All Career Paths"])

with tab1:
    with st.form("predict_form"):

        st.subheader("Skill Ratings")
        c1, c2 = st.columns(2)
        with c1:
            lq = st.slider("Logical Quotient Rating (1–9)", 1, 9, 5)
            cs = st.slider("Coding Skills Rating (1–9)", 1, 9, 5)
        with c2:
            hk = st.slider("Hackathons Participated (0–6)", 0, 6, 2)
            ps = st.slider("Public Speaking Points (1–9)", 1, 9, 5)

        st.divider()
        st.subheader("Learning & Work Style")
        c3, c4 = st.columns(2)
        with c3:
            sl = st.radio("Self-Learning Capability", ["yes", "no"], horizontal=True)
            ec = st.radio("Extra Courses Done", ["yes", "no"], horizontal=True)
            ti = st.radio("Taken Inputs from Seniors", ["yes", "no"], horizontal=True)
        with c4:
            wt = st.radio("Worked in Teams", ["yes", "no"], horizontal=True)
            iv = st.radio("Introverted", ["yes", "no"], horizontal=True)
            ws = st.radio("Work Style", ["hard worker", "smart worker"], horizontal=True)

        st.divider()
        st.subheader("Abilities")
        c5, c6, c7 = st.columns(3)
        with c5:
            mt = st.radio("Management or Technical", ["Management", "Technical"], horizontal=True)
        with c6:
            rw = st.select_slider("Reading & Writing Skills", ["poor", "medium", "excellent"], value="medium")
        with c7:
            mc = st.select_slider("Memory Capability", ["poor", "medium", "excellent"], value="medium")

        st.divider()
        st.subheader("Interests")
        c8, c9 = st.columns(2)
        with c8:
            isub = st.selectbox("Interested Subject", sorted([
                'Computer Architecture','IOT','Management','Software Engineering',
                'cloud computing','data engineering','hacking','networks',
                'parallel computing','programming']))
            icar = st.selectbox("Interested Career Area", sorted([
                'Business process analyst','cloud computing','developer',
                'security','system developer','testing']))
            cert = st.selectbox("Certification Done", sorted([
                'app development','distro making','full stack','hadoop',
                'information security','machine learning','python',
                'r programming','shell programming']))
        with c9:
            wksp = st.selectbox("Workshop Attended", sorted([
                'cloud computing','data science','database security',
                'game development','hacking','system designing','testing','web technologies']))
            comp = st.selectbox("Preferred Company Type", sorted([
                'BPA','Cloud Services','Finance','Product based','SAaS services',
                'Sales and Marketing','Service Based',
                'Testing and Maintainance Services','Web Services','product development']))
            book = st.selectbox("Favourite Type of Books", sorted([
                'Action and Adventure','Anthology','Art','Autobiographies','Biographies',
                'Childrens','Comics','Cookbooks','Diaries','Dictionaries','Drama',
                'Encyclopedias','Fantasy','Guide','Health','History','Horror','Journals',
                'Math','Mystery','Poetry','Prayer books','Religion-Spirituality','Romance',
                'Satire','Science','Science fiction','Self help','Series','Travel','Trilogy']))

        st.divider()
        submitted = st.form_submit_button("🎯 Predict Career", use_container_width=True)

    if submitted:
        try:
            art = artifacts
            NUM = ['Logical quotient rating','hackathons','coding skills rating','public speaking points']
            raw = {
                'Logical quotient rating': lq, 'hackathons': hk,
                'coding skills rating': cs, 'public speaking points': ps,
                'self-learning capability?': sl, 'Extra-courses did': ec,
                'Taken inputs from seniors or elders': ti, 'worked in teams ever?': wt,
                'Introvert': iv, 'reading and writing skills': rw,
                'memory capability score': mc, 'Management or Technical': mt,
                'hard/smart worker': ws, 'certifications': cert,
                'workshops': wksp, 'Interested subjects': isub,
                'interested career area': icar,
                'Type of company want to settle in?': comp,
                'Interested Type of Books': book,
            }
            row = {}
            for col in NUM:
                row[col] = raw[col]
            for col in art['binary_cols']:
                row[col] = 1 if raw[col] == 'yes' else 0
            for col, mapping in art['ordinal_map'].items():
                row[col] = mapping[raw[col]]
            for col in art['nominal_cols']:
                le  = art['label_encoders'][col]
                val = str(raw[col]).strip()
                row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

            df_in      = pd.DataFrame([row])[art['feature_cols']]
            df_in[NUM] = art['scaler'].transform(df_in[NUM])
            pred_idx   = art['model'].predict(df_in.values.astype(np.float32))[0]
            pred_role  = art['target_encoder'].inverse_transform([pred_idx])[0]
            rec        = RECS.get(pred_role, {})

            st.success(f"### {rec.get('icon','')}  Predicted Career:  **{pred_role}**")
            st.write(rec.get('description', ''))
            st.write(f"💰 **Salary Range:** {rec.get('salary')}  &nbsp;|&nbsp;  📈 **Growth:** {rec.get('growth')}")

            st.divider()
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### 📚 Recommended Courses")
                for i, c in enumerate(rec.get('courses', []), 1):
                    st.write(f"{i}. {c}")

                st.markdown("#### 🏅 Certifications to Target")
                for c in rec.get('certs', []):
                    st.write(f"✓ {c}")

            with col_b:
                st.markdown("#### 🛠️ Skills to Build")
                for s in rec.get('skills', []):
                    st.write(f"• {s}")

                st.markdown("#### 📊 Your Skill Profile")
                mm   = {'poor': 3, 'medium': 6, 'excellent': 9}
                cats = ['Logical','Coding','Speaking','Hackathons','Memory','Reading']
                vals = [lq, cs, ps, min(hk*1.5+1, 9), mm.get(mc,5), mm.get(rw,5)]
                fig  = go.Figure(go.Scatterpolar(
                    r=vals+[vals[0]], theta=cats+[cats[0]],
                    fill='toself',
                    line=dict(color='#636EFA', width=2),
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,9])),
                    showlegend=False,
                    margin=dict(l=30,r=30,t=20,b=20),
                    height=280,
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.subheader("All 12 Career Paths")
    st.write("Expand any career to see recommended courses, skills, and certifications.")
    st.divider()

    for role, data in RECS.items():
        with st.expander(f"{data['icon']}  {role}  —  {data['salary']}  |  Growth: {data['growth']}"):
            st.write(f"**{data['description']}**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Recommended Courses**")
                for c in data['courses']:
                    st.write(f"• {c}")
            with col2:
                st.markdown("**Skills to Build**")
                for s in data['skills']:
                    st.write(f"• {s}")
                st.markdown("**Certifications**")
                for c in data['certs']:
                    st.write(f"✓ {c}")
