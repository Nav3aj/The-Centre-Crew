import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- LOAD ISOT ----------
print("Loading ISOT dataset...")
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")
fake_df['label'] = 0
real_df['label'] = 1
print(f" {len(fake_df)} fake, {len(real_df)} real articles loaded")

# ══════════════════════════════════════════════════════
# EXTRA REAL NEWS — covers Indian, global, sports,
# science, politics, economy, health, weather
# ══════════════════════════════════════════════════════
extra_real = [
    # Indian Government & Politics
    "The Reserve Bank of India raised the repo rate by 25 basis points according to an official statement from the Monetary Policy Committee citing persistent inflationary pressures in food and fuel sectors",
    "The Supreme Court of India upheld the constitutional validity of the EWS reservation in a three to two majority ruling stating the quota does not violate the basic structure of the Constitution",
    "Parliament passed the Digital Personal Data Protection Bill with a majority vote on Thursday mandating companies obtain explicit consent from users before collecting personal data with penalties for violations",
    "The Election Commission of India announced the schedule for state assembly elections confirming polling dates security arrangements and the number of constituencies according to an official press release",
    "The Cabinet approved a new scheme to provide financial assistance to farmers affected by drought conditions according to an official press note from the Ministry of Agriculture",
    "The Prime Minister addressed both houses of Parliament on economic growth citing official GDP data from the Central Statistics Office showing improvement in manufacturing and services sectors",
    "Lok Sabha passed the new criminal procedure code with a two thirds majority after extensive debate among members according to the official parliamentary records and gazette notification",
    "The Finance Minister presented the Union Budget in Parliament citing revenue projections infrastructure spending and fiscal deficit targets based on official economic data and ministry reports",
    "The Chief Justice delivered a landmark ruling on privacy rights citing constitutional provisions and international precedents according to court documents filed with the Supreme Court registry",
    "The government spokesperson confirmed implementation of the new education policy following approval from the cabinet committee with details published in the official gazette notification",

    # ISRO & Science
    "ISRO successfully launched the PSLV mission from Sriharikota according to an official press release from the space agency confirming the satellite was placed in the intended orbit",
    "ISRO confirmed the Chandrayaan mission successfully entered lunar orbit according to data received at the Mission Operations Complex in Bengaluru scientists confirmed all systems performing normally",
    "Scientists at the Indian Institute of Science published research on renewable energy in a peer reviewed international journal confirming new solar cell efficiency records achieved in laboratory conditions",
    "The Defence Research and Development Organisation confirmed successful test of the new missile system at the Integrated Test Range meeting all flight objectives according to official ministry statement",
    "Indian scientists published findings in the journal Nature confirming a new species discovered in the Western Ghats the research was conducted over three years with data from multiple field studies",

    # Economy & Finance
    "India retail inflation eased to a four month low according to data released by the Ministry of Statistics driven by lower food prices particularly vegetables the official report confirmed",
    "The stock market closed higher as investors responded positively to strong quarterly earnings from major companies exchange data showed the benchmark index gaining over one percent on high volumes",
    "India GDP growth projected at six point eight percent for the fiscal year per World Bank analysis citing strong domestic consumption resilient services sector and increased government capital expenditure",
    "The Finance Ministry released quarterly data showing improvement in direct tax collections which grew by eighteen percent compared to the same period last year according to official revenue department figures",
    "The RBI governor announced new guidelines for digital lending platforms in a press conference citing concerns about consumer protection and data privacy according to the central bank official statement",
    "Export figures released by the Commerce Ministry showed growth of twelve percent in merchandise exports driven by engineering goods pharmaceuticals and textiles according to official trade data",
    "The government announced production linked incentive scheme for semiconductor manufacturing with approved funding according to the Ministry of Electronics and Information Technology official notification",

    # Health & Medicine
    "The World Health Organisation published updated guidelines recommending new treatment protocols for antimicrobial resistance the study was peer reviewed by researchers from five international universities",
    "The Health Ministry confirmed the nationwide vaccination drive has administered over two billion doses according to official data from the Co-WIN portal cited in the ministry press release",
    "Scientists published clinical trial results in the Lancet showing the new drug reduced recovery time by forty percent the trial involved three thousand patients across twelve hospitals and was independently reviewed",
    "The Indian Council of Medical Research published findings on the prevalence of non communicable diseases based on a nationwide survey of over one hundred thousand households conducted by trained health workers",
    "A new study published in the British Medical Journal by researchers at AIIMS confirmed that regular physical activity significantly reduces the risk of cardiovascular disease based on ten year follow up data",
    "The Food Safety and Standards Authority of India issued new regulations on food labelling according to an official gazette notification requiring manufacturers to display nutritional information clearly",

    # Weather & Environment
    "The India Meteorological Department issued a yellow alert warning of heavy rainfall and thunderstorms with wind speeds reaching fifty kilometres per hour over coastal Karnataka according to the official weather bulletin",
    "The IMD forecast above normal temperatures across northwest India for the coming week according to the official seasonal outlook released by the Meteorological Centre in New Delhi",
    "Scientists from the Indian Institute of Tropical Meteorology published research confirming rising sea surface temperatures in the Arabian Sea the study used forty years of satellite data and ocean buoy measurements",
    "The Central Pollution Control Board released the annual air quality report showing improvement in PM2.5 levels in twelve major cities based on data from continuous ambient air quality monitoring stations",
    "The National Disaster Management Authority issued guidelines for cyclone preparedness following the India Meteorological Department forecast of a depression forming in the Bay of Bengal according to official advisories",

    # Sports
    "The Board of Control for Cricket in India announced the schedule for the upcoming Test series confirmed venues match dates and ticket availability according to the official BCCI press release",
    "Indian athletes won three gold medals at the Commonwealth Games according to official results published by the Games organising committee confirming the country finished in the top five overall",
    "The Athletics Federation of India confirmed the national record in the four hundred metre hurdles was broken at the national championship according to official timing data reviewed by technical officials",
    "FIFA confirmed India will host matches during the World Cup according to an official letter from the federation to the All India Football Federation confirming venue approvals and match schedule",
    "The Olympic Committee confirmed the Indian contingent of over two hundred athletes will participate in the Summer Games according to an official statement from the Indian Olympic Association",

    # International News
    "The United Nations Security Council passed a resolution on climate finance with fifteen member states voting in favour according to official UN documents and confirmed by the Secretary General spokesperson",
    "The World Trade Organisation released its annual trade forecast projecting global merchandise trade growth of two point four percent according to economists cited in the official WTO report",
    "NASA confirmed the James Webb Space Telescope captured new images of a galaxy twelve billion light years away according to data released by the Space Telescope Science Institute peer reviewed by astronomers",
    "The International Monetary Fund revised India growth forecast upward to six point five percent citing strong domestic demand and robust services exports according to the World Economic Outlook report",
    "Ukraine reported continued shelling in the eastern regions according to official statements from the defence ministry confirmed by international observers from the OSCE monitoring mission",
    "The European Central Bank raised interest rates by twenty five basis points according to the official decision published following the governing council meeting citing persistent inflationary pressures across the eurozone",
    "The G20 summit concluded with member nations signing a joint declaration on sustainable development goals according to official communique released by the host country presidency",
    "Iran announced restrictions at the Strait of Hormuz citing ongoing regional tensions the announcement was confirmed by official government statements and reported by Reuters and Al Jazeera",

    # Technology
    "Google announced new artificial intelligence features for its search engine in an official blog post confirmed by the company spokesperson at a press conference attended by technology journalists",
    "The Ministry of Electronics and Information Technology launched the new digital infrastructure initiative according to an official press release confirming investment targets and implementation timeline",
    "Apple reported record quarterly revenue of ninety billion dollars in its earnings call citing strong iPhone sales in India and emerging markets according to official financial results filed with the SEC",
    "The Telecom Regulatory Authority of India released new spectrum auction guidelines according to an official notification published in the gazette inviting comments from industry stakeholders",

    # Shorter headlines (trains model on brief real news too)
    "RBI holds repo rate steady at six point five percent MPC vote unanimous",
    "ISRO Gaganyaan test flight successful all systems nominal says space agency",
    "Supreme Court upholds EWS quota ruling constitutional bench confirms",
    "India wins gold at Asian Games athletics team sets new national record",
    "Parliament passes new criminal law code after three day debate",
    "IMD issues red alert heavy rain warning for Kerala coastal districts",
    "Sensex rises four hundred points on strong quarterly earnings data",
    "WHO recommends updated vaccine formulation for upcoming flu season",
    "India signs trade agreement with UAE boosting bilateral commerce",
    "NASA discovers water ice deposits on lunar south pole confirms findings",
    "UK body confirms Iranian gunboats fired on tanker in Strait of Hormuz crew safe",
    "Two Iranian gunboats fire on tanker in Hormuz crew safe UK maritime authority confirms",
    "Russian forces launch missile strikes on Ukrainian infrastructure Kyiv confirms attack",
    "Fed holds interest rates steady citing cooling inflation data official statement",
    "China GDP growth slows to four point six percent official statistics bureau confirms",
    "Pakistan floods displace millions government declares national emergency",
    "Bangladesh elections held peacefully voter turnout at sixty two percent commission confirms",
    "Saudi Arabia cuts oil production by one million barrels per day OPEC confirms",
    "Japan earthquake magnitude six point two strikes northern region no major damage reported",
    "Australia cricket team wins the Ashes series three to one official result confirmed",
]

# ══════════════════════════════════════════════════════
# EXTRA FAKE NEWS — diverse fake patterns
# ══════════════════════════════════════════════════════
extra_fake = [
    "SHOCKING secret cure doctors don't want you to know miracle treatment banned by government exposed by whistleblower share before deleted",
    "BREAKING anonymous insiders reveal government secretly planning to ban all cash wake up people mainstream media hiding truth deep state agenda",
    "URGENT 5G towers are secretly emitting radiation that destroys human DNA thousands dying and the media is completely silent share before deleted",
    "MIRACLE Indian spice cures cancer diabetes heart disease in seven days big pharma furious hidden secret they suppressed for decades finally revealed",
    "EXPOSED politician caught secret deal to sell country to foreign power anonymous sources confirm the shocking betrayal mainstream media is covering up",
    "WARNING vaccines contain microchips to track every citizen whistleblower leaks classified documents proving the global conspiracy you won't believe",
    "BREAKING scientists discover moon landing was completely staged by NASA in Hollywood studio secret documents leaked by anonymous government insider",
    "URGENT all bank accounts will be frozen tomorrow anonymous source from inside the ministry reveals shocking plan mainstream media completely silent",
    "SHOCKING secret cancer cure suppressed by big pharma for twenty years finally exposed by brave doctor share this before they take it down",
    "MIRACLE water charged with crystals cures all diseases doctors hiding this secret from public share before big pharma gets this deleted forever",
    "EXPOSED the government is poisoning the water supply with dangerous chemicals to control the population whistleblower reveals shocking truth share now",
    "BREAKING deep state plot to steal the election exposed by brave whistleblower leaked documents confirm massive conspiracy mainstream media ignoring it",
    "This one weird trick burns belly fat in three days without exercise doctors hate this secret that big pharma has been hiding for years",
    "BOMBSHELL anonymous insider from the PMO confirms prime minister secretly signed deal to give national assets to foreign billionaires wake up India",
    "Scientists discover that eating this common fruit every day instantly cures high blood pressure diabetes and cancer doctors are furious about this",
    "EXPOSED illuminati plan to reduce world population using vaccines and 5G towers anonymous insider leaks classified documents share before censored",
    "URGENT warning the rupee will collapse within forty eight hours government hiding the truth anonymous financial insider reveals shocking information",
    "BREAKING bombshell video shows top minister accepting bribe anonymous source confirms massive corruption mainstream media is completely ignoring this",
    "MIRACLE herb discovered in Himalayas cures all known diseases scientists baffled big pharma trying to suppress this ancient secret cure revealed",
    "SHOCKING truth about what really happened government covering up the real story anonymous insiders reveal what mainstream media won't tell you ever",
]

extra_real_df = pd.DataFrame({'text': extra_real * 60, 'label': 1})
extra_fake_df = pd.DataFrame({'text': extra_fake * 60, 'label': 0})

# ---------- COMBINE ----------
df = pd.concat([fake_df, real_df, extra_real_df, extra_fake_df], ignore_index=True)

if 'title' in df.columns:
    df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
else:
    df['combined'] = df['text'].fillna('')

df['combined'] = df['combined'].apply(clean_text)
df = df[df['combined'].str.len() > 10]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n Total rows : {len(df)}")
print(f"FAKE (0)      : {(df['label']==0).sum()}")
print(f"REAL (1)      : {(df['label']==1).sum()}\n")

X = df['combined']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print("🏋️ Training ensemble model...")
lr     = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', class_weight='balanced')
rf     = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
gboost = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

voting = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gboost', gboost)],
    voting='soft', n_jobs=-1
)
voting.fit(X_train_vec, y_train)

y_pred = voting.predict(X_test_vec)
print(f"\n Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['FAKE','REAL']))

joblib.dump(voting,     "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("model.pkl and vectorizer.pkl saved!")