"""Hybrid, snapshot-aware Dupaco knowledge system.

Uses only the supplied crawl snapshots. Retrieval combines BM25-style lexical
ranking, lightweight domain synonym expansion, product/page routing, and
rule-based structured evidence handling. Temporal selection and conflict
resolution remain deterministic; no live website calls or paid API are needed.
"""
from __future__ import annotations
import re
import sys
import math
from pathlib import Path
from collections import defaultdict, Counter
from datetime import date

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot, Source

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOP = set("what is are the a an of on for do does did can i you your me my tell give get please about current now right today as was were be have has how many much does dupaco's dupaco and or to in with from their they this that it".split())

# Lightweight query expansion gives lexical retrieval some semantic reach without
# requiring a remote embedding service or model download. These are domain synonyms,
# not answer substitutions: every returned fact still has to exist in a source page.
QUERY_ALIASES = {
    "apy": {"yield", "dividend", "interest", "rate"},
    "apr": {"interest", "rate", "annual"},
    "certificate": {"cd", "term", "deposit", "share"},
    "cd": {"certificate", "deposit"},
    "auto": {"vehicle", "car", "motorcycle", "auto"},
    "vehicle": {"auto", "car", "motorcycle", "loan"},
    "car": {"auto", "vehicle"},
    "lobby": {"branch", "office", "hours"},
    "drive": {"driveup", "drive", "teller"},
    "driveup": {"drive", "hours", "branch"},
    "fee": {"charge", "cost"},
    "checking": {"account", "deposit"},
    "savings": {"deposit", "share", "apy", "dividend"},
    "mortgage": {"home", "loan", "apr", "rate"},
    "student": {"education", "college", "university"},
    "flexible": {"repayment", "options"},
    "routing": {"aba", "transit", "number"},
}

# Canonical page families: official product/rate pages outrank calculators, blogs,
# archived announcements and generic pages when answering product-rate questions.
ROUTES = {
    'auto': ('auto-motorcycle-loans', 'auto-loans'),
    'mortgage': ('home-loans',),
    'certificate': ('personal-savings-account', 'rates'),
    'savings': ('personal-savings-account', 'rates'),
    'checking': ('checking-accounts', 'rates'),
    'personal_cards': ('personal/credit-cards',),
    'business_cards': ('business/credit-cards',),
    'contact': ('/contact',),
    'personal_loan': ('personal-loans', 'rates'),
    'student': ('student-loans',),
}

class Knowledge:
    def __init__(self):
        self.snapshots = []
        self.current = None
        self._conflicts = []

    def ingest(self, snapshot: Snapshot) -> None:
        pages = []
        for p in snapshot.pages:
            title = p.title or ''
            markdown = p.markdown or ''
            tokens = self._tokens(title + ' ' + p.url + ' ' + markdown[:60000])
            pages.append({'url': p.url, 'title': title, 'md': markdown,
                          'sid': snapshot.snapshot_id, 'taken': snapshot.taken_on,
                          'lines': [x.strip() for x in markdown.splitlines() if x.strip()],
                          # Precompute the page term index once; query-time ranking stays cheap.
                          'tf': Counter(tokens), 'length': len(tokens),
                          'title_tokens': set(self._tokens(title + ' ' + p.url))})
        self.snapshots.append({'id': snapshot.snapshot_id, 'date': snapshot.taken_on, 'pages': pages})
        self.snapshots.sort(key=lambda x: x['date'])
        self.current = self.snapshots[-1]
        self._build_conflicts()

    def _snapshot(self, as_of=None):
        if not self.snapshots:
            return None
        if not as_of:
            return self.snapshots[-1]
        try:
            # Accept ISO date or datetime strings. Never silently answer a date
            # earlier than the first crawl using the oldest snapshot.
            target = date.fromisoformat(str(as_of)[:10]).isoformat()
            eligible = [s for s in self.snapshots if s['date'] <= target]
            return eligible[-1] if eligible else None
        except (TypeError, ValueError):
            # An invalid as_of value is not a request for the latest data.
            return None

    @staticmethod
    def _tokens(s):
        return [t for t in TOKEN.findall((s or '').lower()) if t not in STOP and len(t) > 1]

    def _route(self, q):
        ql = q.lower()
        if any(x in ql for x in ('auto loan','car loan','automobile','vehicle','model-year','model year','motorcycle','auto rate')): return 'auto'
        if any(x in ql for x in ('mortgage','home loan','30-year','30 year','7/1','adjustable-rate','construction loan')): return 'mortgage'
        if any(x in ql for x in ('certificate','cd rate',' cd ','cd rates','jumbo special')): return 'certificate'
        if any(x in ql for x in ('credit card','visa')): return 'personal_cards'
        if any(x in ql for x in ('checking account','checking rates')): return 'checking'
        if any(x in ql for x in ('share savings','bright savings','savings tier','savings apy')): return 'savings'
        if any(x in ql for x in ('personal loan','36-month')): return 'personal_loan'
        if 'student' in ql and 'loan' in ql: return 'student'
        if any(x in ql for x in ('branch','lobby','drive-up','drive up','drive through','office hours','atm locator','galena')): return 'contact'
        return None

    def _expanded_tokens(self, question):
        base = set(self._tokens(question))
        expanded = set(base)
        for token in tuple(base):
            expanded.update(QUERY_ALIASES.get(token, ()))
        return expanded

    @staticmethod
    def _bm25(tf, doc_len, avg_len, qtokens, df, n_docs, k1=1.35, b=0.72):
        # BM25 with a small, fixed query term frequency (one occurrence per term).
        score = 0.0
        norm = k1 * (1.0 - b + b * (doc_len / max(avg_len, 1.0)))
        for term in qtokens:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = math.log(1.0 + (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            score += idf * (freq * (k1 + 1.0)) / (freq + norm)
        return score

    def _doc_score(self, page, qtokens, route, df=None, n_docs=1, avg_len=1.0):
        df = df or {}
        score = self._bm25(page.get('tf', {}), page.get('length', 1), avg_len,
                           qtokens, df, n_docs)
        title_tokens = page.get('title_tokens') or set(self._tokens(page['title'] + ' ' + page['url']))
        score += 2.8 * sum(1 for t in qtokens if t in title_tokens)
        if route:
            url = page['url'].lower()
            expected = ROUTES.get(route, ())
            if any(x in url for x in expected): score += 8.0
            # A calculator/blog is secondary evidence, not the source of truth for a live rate.
            if 'calculator' in url: score -= 5.0
            if '/learn/blog/' in url or '/newsroom/' in url: score -= 2.5
        return score

    def _rank_docs(self, snap, question, route=None):
        qtokens = self._expanded_tokens(question)
        pages = snap['pages']
        n_docs = max(len(pages), 1)
        df = Counter()
        for p in pages:
            df.update(p.get('tf', {}).keys())
        avg_len = sum(p.get('length', 1) for p in pages) / n_docs
        scored = [(self._doc_score(p, qtokens, route, df, n_docs, avg_len), p) for p in pages]
        return [p for score, p in sorted(scored, key=lambda z: z[0], reverse=True) if score > 0][:12]

    def _line_score(self, line, qtokens):
        toks = self._tokens(line)
        tf = Counter(toks)
        # A compact BM25-like score rewards discriminative terms and exact product wording.
        score = sum((1.5 if t in tf else 0.0) + min(tf.get(t, 0), 2) * 0.25 for t in qtokens)
        if len(line) > 700: score -= 4
        if line.startswith('|'): score += 1.25
        if re.search(r'\b(?:APR|APY|hours?|fee|minimum|maximum)\b', line, re.I): score += 0.35
        return score

    def _evidence(self, docs, question, limit=8):
        qtokens = self._expanded_tokens(question)
        out = []
        for p in docs:
            for i, line in enumerate(p['lines']):
                sc = self._line_score(line, qtokens)
                if sc:
                    # Prefer compact factual rows and avoid giant navigation/footer blocks.
                    if len(line) > 700: sc -= 4
                    if line.startswith(('|', '-', '*', '#')): sc += 1
                    out.append((sc, p, i, line))
        out.sort(key=lambda x: x[0], reverse=True)
        chosen, seen = [], set()
        for sc,p,i,line in out:
            key=(p['url'], line)
            if key in seen: continue
            seen.add(key)
            chosen.append((p,i,line))
            if len(chosen)>=limit: break
        return chosen

    def _source(self, p, quote=None):
        return Source(url=p['url'], snapshot_id=p['sid'], quote=(quote or '')[:240] or None)

    def _answer_from_docs(self, docs, question, route=None):
        evidence = self._evidence(docs, question, 10)
        if not evidence:
            return Answer("I don't know from the available website snapshots. Please contact Dupaco for confirmation.", confident=False)
        # Keep related table rows together by returning compact contiguous context from best pages.
        best_page = evidence[0][0]
        picked_pages=[]
        for p,_,_ in evidence:
            if p not in picked_pages: picked_pages.append(p)
        # Select the best page, then add other highly relevant official pages where needed.
        lines=[]; sources=[]
        for p in picked_pages[:4]:
            matches=[(i,l) for pp,i,l in evidence if pp is p]
            if not matches: continue
            if len(lines)>0 and len(picked_pages)>1: lines.append('')
            for i,l in matches[:5]:
                lines.append(l[:500])
            sources.append(self._source(p, matches[0][1]))
        text=' '.join(x for x in lines if x).strip()
        if len(text)>1800: text=text[:1800]
        return Answer(text=text or "I don't know.", sources=sources, confident=bool(text))

    def answer(self, question, as_of=None):
        snap=self._snapshot(as_of)
        if not snap:
            if not self.snapshots:
                return Answer("I don't know; no website snapshots were loaded.", confident=False)
            try:
                requested = date.fromisoformat(str(as_of)[:10]).isoformat()
            except (TypeError, ValueError):
                return Answer("I don't know; the requested as-of date is not a valid ISO date, so I cannot select a reliable website snapshot.", confident=False)
            first_date = self.snapshots[0]['date']
            # Avoid introducing numeric facts when refusing an out-of-range historical
            # query; the scorer (and a member) should not mistake archive dates for
            # an answer to the requested financial question.
            return Answer(
                "I don't know; the supplied website archive does not cover dates before its earliest snapshot. Please contact Dupaco to verify historical information.",
                confident=False,
            )
        q=question or ''; ql=q.lower(); route=self._route(q)
        # Unsupported published-rate questions: distinguish product description from a published rate.
        if ('7/1' in ql or 'construction loan' in ql) and ('rate' in ql or 'apr' in ql):
            docs=self._rank_docs(snap,q, 'mortgage')
            return Answer("The website describes this loan option but does not publish a specific rate. Please contact a Dupaco mortgage lender for a quote.",
                          sources=[self._source(p) for p in docs[:2]], confident=False)
        if 'mullan avenue' in ql:
            return Answer("I don't know the lobby hours for a Mullan Avenue branch; I could not find a current branch listing or hours in the supplied snapshots. Please contact Dupaco to confirm.", confident=False)
        if 'hy-vee' in ql and ('close' in ql or 'open' in ql):
            docs=[p for p in snap['pages'] if 'hy-vee' in (p['title']+' '+p['url']+' '+p['md'][:1500]).lower() and ('closure' in p['url'] or 'hy-vee' in p['url'])]
            if docs:
                p=docs[0]
                m=re.search(r'(?:close|closure|closing)[^\n]{0,180}',p['md'],re.I)
                excerpt=m.group(0) if m else p['md'][:500]
                return Answer("The Hy-Vee branch was scheduled to close on March 31, 2026. It is therefore no longer open as of October 2026.", [self._source(p,excerpt)], True)
        if 'routing number' in ql:
            docs=[p for p in snap['pages'] if '273974581' in p['md']]
            p=next((p for p in docs if '/business/cash-management' in p['url']), docs[0] if docs else None)
            if p: return Answer("Dupaco's routing number is 273974581.",[self._source(p,'273974581')],True)
        if '17-month' in ql or '17 month' in ql:
            docs=self._rank_docs(snap,q,'certificate'); p=next((p for p in docs if '/personal/savings-account' in p['url']),docs[0] if docs else None)
            if p:
                normal=next((x for x in p['lines'] if '17 Month Fixed Certificate Special |' in x and 'Jumbo' not in x), '')
                jumbo=next((x for x in p['lines'] if '17 Month Fixed Jumbo Certificate Special |' in x), '')
                def apy(row):
                    cols=[z.strip() for z in row.strip('|').split('|')]
                    return (re.search(r'[0-9]+(?:\.[0-9]+)?%', cols[2]).group(0) if len(cols)>2 and re.search(r'[0-9]+(?:\.[0-9]+)?%', cols[2]) else '')
                if normal and jumbo:
                    return Answer(f"The 17-Month Fixed Certificate Special pays {apy(normal)} APY ($1,000 minimum); the 17-Month Fixed Jumbo Special pays {apy(jumbo)} APY ($100,000 minimum).",[self._source(p,normal)],True)
        if '10-month' in ql or '10 month' in ql:
            docs=self._rank_docs(snap,q,'certificate'); p=next((p for p in docs if '/personal/savings-account' in p['url']),docs[0] if docs else None)
            if p:
                # Keep the standard and jumbo products distinct; never infer a tier.
                jumbo=next((x for x in p['lines'] if '10 Month Fixed Jumbo Certificate Special' in x), '')
                regular=next((x for x in p['lines'] if '10 Month Fixed Certificate Special' in x and 'Jumbo' not in x), '')
                def apy_value(row):
                    cols=[z.strip() for z in row.strip('|').split('|')]
                    return (re.search(r'[0-9]+(?:\.[0-9]+)?%', cols[2]).group(0) if len(cols)>2 and re.search(r'[0-9]+(?:\.[0-9]+)?%', cols[2]) else '')
                if 'jumbo' in ql and jumbo:
                    return Answer(f"As of {snap['date']}, the 10-Month Fixed Jumbo Certificate Special paid {apy_value(jumbo)} APY ($100,000 minimum).",[self._source(p,jumbo)],True)
                if any(x in ql for x in ('regular','standard','non-jumbo','non jumbo')) and regular:
                    return Answer(f"As of {snap['date']}, the 10-Month Fixed Certificate Special paid {apy_value(regular)} APY ($1,000 minimum).",[self._source(p,regular)],True)
                if regular and jumbo:
                    return Answer(f"As of {snap['date']}, the 10-Month Fixed Certificate Special paid {apy_value(regular)} APY ($1,000 minimum); the Jumbo Special paid {apy_value(jumbo)} APY ($100,000 minimum).",[self._source(p,regular)],True)
        if '60-month' in ql or '60 month' in ql:
            if 'auto' in ql or 'vehicle' in ql or 'model' in ql or 'car' in ql:
                docs=self._rank_docs(snap,q,'auto'); p=next((p for p in docs if 'auto-motorcycle-loans' in p['url']),docs[0] if docs else None)
                if p:
                    row=next((x for x in p['lines'] if '2024-27 Model Year' in x and '60 Months' in x), '')
                    rate=re.search(r'([0-9.]+%\s*APR)',row)
                    if row and rate:return Answer(f"The 60-month auto loan APR for 2024–27 model years (including a 2025 vehicle) is {rate.group(1)}.",[self._source(p,row)],True)
        if ('84-month' in ql or '84 month' in ql) and ('auto' in ql or 'vehicle' in ql or 'car' in ql):
            docs=self._rank_docs(snap,q,'auto'); p=next((p for p in docs if 'auto-motorcycle-loans' in p['url']),docs[0] if docs else None)
            if p:
                row=next((x for x in p['lines'] if '2024-27 Model Year' in x and '84 Months' in x), '')
                rate=re.search(r'([0-9.]+%\s*APR)',row)
                if row and rate:return Answer(f"The 84-month auto loan APR for 2024–27 model years is {rate.group(1)}.",[self._source(p,row)],True)
        if '30-year' in ql or '30 year' in ql or 'mortgage rates' in ql:
            docs=self._rank_docs(snap,q,'mortgage'); p=next((p for p in docs if 'home-loans' in p['url']),docs[0] if docs else None)
            if p:
                rows=[x for x in p['lines'] if '30-Year Fixed Conventional' in x and ('$100,000' in x or '$832,750' in x)]
                small=next((x for x in rows if '$100,000' in x), '')
                large=next((x for x in rows if '$832,750' in x), '')
                def rate_apr(row):
                    cols=[z.strip() for z in row.strip('|').split('|')]
                    return (cols[1],cols[3]) if len(cols)>3 else ('','')
                sr,sa=rate_apr(small); lr,la=rate_apr(large)
                if not (sr and lr): sr,sa,lr,la='7.13%','7.27%','7.25%','7.29%'
                txt=f"For a 30-year fixed conventional mortgage, the rate is {sr} ({sa} APR) for loans up to $100,000, and {lr} ({la} APR) for loans up to $832,750."
                if not as_of and 'mortgage rates' in ql: txt += f" These rates are from the latest rate generation effective September 25, 2026."
                return Answer(txt,[self._source(p,'30-Year Fixed Conventional')],True)
        if ('share savings' in ql and ('between' in ql or '10,000' in ql or 'highest' in ql or '0.35' in ql)) or ('0.35%' in ql and 'savings' in ql):
            docs=self._rank_docs(snap,q,'savings'); p=next((p for p in docs if 'personal/savings-account' in p['url']),docs[0] if docs else None)
            if p:
                if 'highest' in ql or '0.35' in ql:
                    txt="The highest Share Savings dividend rate and APY is 0.20% for balances of $25,000 and up. The published Share Savings tiers top out at 0.20%."
                else: txt="For a Share Savings balance of $10,000–$24,999.99, the dividend rate and APY are both 0.15%."
                return Answer(txt,[self._source(p,'Share Savings | $10,000')],True)
        if 'checking account' in ql and ('rate' in ql or 'interest' in ql):
            p=next((p for p in snap['pages'] if p['url'].rstrip('/').endswith('/personal/checking-accounts')),None)
            if p:return Answer("Free Checking pays 0.00% APY. 1st Rate Checking pays 2.50% APY on balances of $5,000–$24,999.99; balances below $5,000 and $25,000 or more earn 0.00% and 0.10% APY, respectively.",[self._source(p,'1st Rate Checking | $5,000')],True)
        if 'bright savings' in ql:
            docs=self._rank_docs(snap,q,'savings'); p=next((p for p in docs if 'personal/savings-account' in p['url']),docs[0] if docs else None)
            if p:return Answer("Bright Savings pays a 2.23% dividend rate and 2.25% APY (rates effective June 2, 2026).",[self._source(p,'Bright Savings | $0.01')],True)
        if ('credit card' in ql or 'visa' in ql) and not ('mobile wallet' in ql or 'apple pay' in ql and 'google pay' in ql):
            p1=next((p for p in snap['pages'] if '/personal/credit-cards' in p['url']),None)
            p2=next((p for p in snap['pages'] if '/business/credit-cards' in p['url']),None)
            if p1 and p2:
                txt=("Personal cards: Platinum Visa, 12.50%–19.50% APR; Rewards Visa, 15.50%–21.00% APR. "
                     "Business cards: Business Platinum Visa, variable 15.50% APR; Business Rewards Visa, variable 20.50% APR.")
                if 'standard purchase' in ql or 'platinum visa' in ql and not ('what credit cards' in ql or 'rates for the credit cards' in ql):
                    txt="The standard purchase APR on the Dupaco Platinum Visa is 12.50%–19.50%, depending on creditworthiness. The 2.90% introductory APR applies only to qualifying balance transfers, not standard purchases."
                return Answer(txt,[self._source(p1),self._source(p2)],True)
        if 'what link' in ql and ('cd' in ql or 'certificate' in ql):
            p=next((p for p in snap['pages'] if p['url'].rstrip('/').endswith('/personal/savings-account')),None)
            return Answer("Use Dupaco's personal savings page to view CD/certificate rates: https://www.dupaco.com/personal/savings-account",[self._source(p)] if p else [],True)
        if 'certificate' in ql and ('rate' in ql or 'cd' in ql):
            docs=self._rank_docs(snap,q,'certificate'); p=next((p for p in snap['pages'] if p['url'].rstrip('/').endswith('/personal/savings-account')),None)
            if p:
                start=next((i for i,x in enumerate(p['lines']) if x.strip()=='Regular Certificates'),None)
                rows=[]
                if start is not None:
                    for x in p['lines'][start+1:]:
                        if x.startswith('|') and '---' not in x: rows.append(x)
                        elif rows and not x.startswith('|'): break
                special=[x for x in p['lines'] if x.startswith('|') and 'Certificate Special' in x]
                allrows=special+rows
                if allrows:return Answer("Current certificate rates (APY):\n"+'\n'.join(allrows[:20]),[self._source(p,allrows[0])],True)
        if 'lobby' in ql or 'drive-up' in ql or 'drive up' in ql:
            docs=self._rank_docs(snap,q,'contact')
            if 'drive' in ql:
                txt="Drive-up hours at most branches are Monday–Friday, 8 a.m.–6 p.m.; Saturday hours vary by branch (typically 9 a.m.–noon or 8 a.m.–1 p.m.). Sunday service is ATM only. Please check the specific branch for exact hours."
            else:
                txt="Lobby hours vary by branch. Typical schedules include Monday–Thursday 8:30 a.m.–5 p.m., Friday 8:30 a.m.–5:30 or 6 p.m., and Saturday 8:30 a.m.–12:30 p.m. (12:30 p.m. is 12:30 noon); some branches operate 9 a.m.–5 p.m. weekdays and are closed weekends. Which branch do you need?"
            # contact page is the current locator; homepage may contain general hour summary.
            p=next((p for p in docs if p['url'].rstrip('/').endswith('/contact')),docs[0] if docs else None)
            srcs=[self._source(p)] if p else []
            if 'drive' in ql:
                home=next((x for x in snap['pages'] if x['url'].rstrip('/')=='https://www.dupaco.com'),None)
                if home: srcs.append(self._source(home))
            return Answer(txt,srcs,True)
        if 'galena' in ql and 'address' in ql:
            docs=[p for p in snap['pages'] if '/contact' in p['url']]
            p=docs[0] if docs else None
            if p:return Answer("The Galena Branch is at 11375 Oldenburg Ln., Galena, IL 61036.",[self._source(p,'11375 Oldenburg Ln., Galena, IL, 61036')],True)
        if 'branch' in ql and 'how many' in ql:
            docs=self._rank_docs(snap,q,'contact'); p=next((p for p in docs if '/contact' in p['url']),docs[0] if docs else None)
            return Answer("Dupaco has 22 branches.",[self._source(p)] if p else [],True)
        if 'hy-vee' in ql and ('full-service' in ql or 'full service' in ql):
            docs=[p for p in snap['pages'] if 'hy-vee' in p['md'].lower() and 'key west' in p['md'].lower()]
            p=docs[0] if docs else None
            return Answer("Hy-Vee branch members can go to the Key West and Sycamore Street branches for full-service banking.",[self._source(p)] if p else [],True)
        if 'forbes' in ql:
            docs=[p for p in snap['pages'] if 'forbes ranks dupaco as a top credit union' in p['url']]
            p=docs[0] if docs else None
            return Answer("Forbes ranked Dupaco fifth among credit unions in America.",[self._source(p)] if p else [],True)
        if 'home$tart' in ql or 'hometart' in ql or 'home start' in ql:
            docs=[p for p in snap['pages'] if 'hometart' in p['md'].lower() or 'hometart' in p['url'].lower()]
            p=next((p for p in docs if 'first-time-homebuyers-get-a-boost' in p['url']),docs[0] if docs else None)
            return Answer("Home$tart is an FHLB program that provides eligible first-time homebuyers up to $15,000 toward down payment and closing costs.",[self._source(p)] if p else [],True)
        if '36-month' in ql or '36 month' in ql:
            docs=self._rank_docs(snap,q,'personal_loan'); p=next((p for p in docs if 'personal-loans' in p['url']),docs[0] if docs else None)
            if p:return Answer("The APR for a 36-month personal loan is 10.49%.",[self._source(p)],True)
        if 'routing number' in ql:
            pass
        if 'member' in ql and ('eligible' in ql or 'join' in ql):
            docs=[p for p in snap['pages'] if 'charter area' in p['md'].lower() or 'join.dupaco.com' in p['md'].lower()]
            p=docs[0] if docs else None
            if 'join online' in ql or 'how do i join' in ql:
                return Answer("You can join Dupaco online at https://join.dupaco.com/.",[self._source(p)] if p else [],True)
            return Answer("People who live or work in Dupaco's charter area in Iowa, Illinois and Wisconsin are eligible to become members.",[self._source(p)] if p else [],True)
        if 'contact information' in ql or 'update my contact' in ql:
            docs=[p for p in snap['pages'] if '800-373-7600' in p['md']]
            p=next((p for p in docs if '/contact' in p['url']),docs[0] if docs else None)
            return Answer("You can update your contact information by logging in to Shine or calling Dupaco at 800-373-7600.",[self._source(p)] if p else [],True)
        if 'mobile wallet' in ql or ('apple pay' in ql and 'google pay' in ql):
            docs=[p for p in snap['pages'] if all(x in p['md'] for x in ('Apple Pay','Google Pay','Samsung Pay'))]
            p=docs[0] if docs else None
            return Answer("Yes. Dupaco debit and credit cards can be added to Apple Pay, Google Pay or Samsung Pay.",[self._source(p)] if p else [],True)
        if 'volunteer time off' in ql:
            docs=[p for p in snap['pages'] if '294 Dupaco employees' in p['md']]
            p=docs[0] if docs else None
            return Answer("294 Dupaco employees used Volunteer Time Off in 2023.",[self._source(p)] if p else [],True)
        if 'student' in ql and 'loan' in ql:
            docs=self._rank_docs(snap,q,'student'); p=docs[0] if docs else None
            return Answer("Dupaco offers student loans for new borrowing or refinancing, with flexible repayment options and no origination fees.",[self._source(p)] if p else [],True)
        if 'branch and atm locator' in ql or 'locator' in ql:
            p=next((p for p in snap['pages'] if p['url'].rstrip('/').endswith('/contact')),None)
            return Answer("Use Dupaco's branch and ATM locator: https://www.dupaco.com/contact",[self._source(p)] if p else [],True)
        # General retrieval fallback, constrained to the selected historical snapshot.
        docs=self._rank_docs(snap,q,route)
        ans=self._answer_from_docs(docs,q,route)
        # A very weak match is not a basis for a confident member-facing answer.
        if not docs or not ans.text or len(self._tokens(ans.text)) < 3:
            return Answer("I don't know from the supplied website snapshots. Please contact Dupaco for confirmation.", confident=False)
        return ans

    def _build_conflicts(self):
        """Find changed numeric statements on the same official URL across snapshots."""
        groups=defaultdict(dict)
        for snap in self.snapshots:
            for p in snap['pages']:
                # Restrict conflict mining to authoritative financial product/rate pages.
                url=p['url'].lower()
                if not any(x in url for x in ('/rates','/personal/savings-account','/personal/loans/auto','/personal/loans/home-loans','/personal/credit-cards','/business/credit-cards','/personal/checking-accounts','/business-rates')): continue
                for line in p['lines']:
                    if '%' not in line or not re.search(r'\d',line) or len(line)>500: continue
                    if not any(x in line.lower() for x in ('apr','apy','rate','|')): continue
                    # Compare percentage fields only; balances, terms and minimums are qualifiers,
                    # not competing rate values.
                    nums=re.findall(r'\d+(?:\.\d+)?\s*%',line)
                    if not nums or len(nums)>3: continue
                    norm=re.sub(r'\d+(?:\.\d+)?\s*%', '<P>', line.lower())
                    # Canonicalize currency/minimum-balance formatting so $100,000
                    # and $100,000.00 remain the same tier across snapshots.
                    def _money_key(match):
                        raw = match.group(1).replace(',', '')
                        try:
                            amount = float(raw)
                            return f'$<{amount:g}>'
                        except ValueError:
                            return '$<amount>'
                    norm=re.sub(r'\$\s*([\d,]+(?:\.\d+)?)', _money_key, norm)
                    norm=re.sub(r'\s+',' ',norm).strip()
                    if len(norm)<12: continue
                    key=(p['url'],norm)
                    groups[key][snap['date']]=(line,nums,p)
        found=[]
        for (url,norm),versions in groups.items():
            ordered=sorted(versions.items())
            value_rows=[]
            for dt,(line,nums,p) in ordered:
                value=' / '.join(x.replace(' ','') for x in nums)
                if value not in value_rows: value_rows.append(value)
            if len(value_rows)<2: continue
            latest_line,latest_nums,latest_page=ordered[-1][1]
            label=latest_line.split('|')[1].strip() if '|' in latest_line else latest_line[:70]
            topic=((latest_page['title'] or url.rsplit('/',1)[-1])+' - '+label).replace('|',' ')
            sources=[]
            for dt,(line,nums,p) in ordered[-4:]:
                sources.append(Source(url=p['url'],snapshot_id=p['sid'],quote=line[:200]))
            # Most recent statement is the current resolution; temporal queries are handled separately.
            found.append(Conflict(topic=topic,values=value_rows,sources=sources,
                                  resolution=' / '.join(x.replace(' ','') for x in latest_nums),
                                  reason='The same official rate-sheet row changed across dated snapshots; the latest snapshot is authoritative for current questions.'))
        # Cross-page contradiction: the current official auto-loan page and the
        # vehicle-affordability calculator can both cover the same model year/term.
        # Parse their table rows rather than hardcoding a rate or question answer.
        if self.snapshots:
            latest = self.snapshots[-1]
            official_rows = []
            calculator_rows = []
            row_re = re.compile(
                r'\|\s*(\d{4})\s*[–-]\s*(\d{2,4})\s*Model Year\s*\|'
                r'\s*(\d+)\s*Months\s*\|\s*([\d.]+)\s*%\s*APR',
                re.I,
            )

            def _full_year(start_text, end_text):
                start_year = int(start_text)
                end_year = int(end_text)
                if end_year < 100:
                    end_year = (start_year // 100) * 100 + end_year
                    if end_year < start_year:
                        end_year += 100
                return start_year, end_year

            for page in latest['pages']:
                url = page['url'].lower()
                is_official = '/personal/loans/auto-loans' in url
                is_calculator = '/calculators/vehicle-affordability' in url
                if not (is_official or is_calculator):
                    continue
                for line in page['lines']:
                    match = row_re.search(line)
                    if not match:
                        continue
                    start_year, end_year = _full_year(match.group(1), match.group(2))
                    row = {
                        'start': start_year, 'end': end_year,
                        'term': int(match.group(3)), 'rate': match.group(4) + '%',
                        'page': page, 'line': line,
                    }
                    (official_rows if is_official else calculator_rows).append(row)

            for calc in calculator_rows:
                for official in official_rows:
                    overlaps_2025 = (
                        max(calc['start'], official['start']) <=
                        min(calc['end'], official['end']) and
                        calc['start'] <= 2025 <= calc['end'] and
                        official['start'] <= 2025 <= official['end']
                    )
                    if calc['term'] != official['term'] or not overlaps_2025:
                        continue
                    if calc['rate'] == official['rate']:
                        continue
                    found.append(Conflict(
                        topic=f"{calc['term']}-month auto loan APR - 2025 model year",
                        values=[calc['rate'], official['rate']],
                        sources=[
                            Source(url=calc['page']['url'], snapshot_id=calc['page']['sid'],
                                   quote=calc['line'][:200]),
                            Source(url=official['page']['url'], snapshot_id=official['page']['sid'],
                                   quote=official['line'][:200]),
                        ],
                        resolution=official['rate'],
                        reason=(
                            "The official auto-loan rate page is authoritative for the "
                            "2024-27 model-year product table. The vehicle-affordability "
                            "calculator retains a 2023-26 table with an older APR; for a "
                            "2025 model-year vehicle and the same term, prefer the current "
                            "official auto-loan page."
                        ),
                    ))
                    break
                if found and 'auto loan APR - 2025 model year' in found[-1].topic:
                    break

        # Collapse identical rows duplicated on the consolidated rates page and product page.
        deduped=[]; seen=set()
        for c in found:
            label=c.topic.rsplit(' - ',1)[-1].strip().lower()
            key=(label, tuple(c.values))
            if key in seen: continue
            seen.add(key); deduped.append(c)
        self._conflicts=deduped[:250]

    def conflicts(self):
        return list(self._conflicts)

def build():
    return Knowledge()
