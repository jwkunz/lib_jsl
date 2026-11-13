#!/usr/bin/env python3
"""Numbers and captions the figures and tables of the book (run from book/src; idempotent).

Figures get "Figure C.k" below them, with the image's alt text as the caption; tables included from
generated/ get "Table C.k" above them, with their column names. C is the chapter number in SUMMARY.md."""
import re,os
root='.'
order=[]
for l in open('SUMMARY.md'):
    m=re.search(r"\]\(([^)]+\.md)\)",l)
    if m and m.group(1) not in order: order.append(m.group(1))
chap={}; n=0
for f in order:
    if '/' not in f: continue
    d=f.split('/')[0]
    if d not in chap: n+=1; chap[d]=n
def splitcells(row):
    cells=[];cur="";inm=False
    row=row.strip()
    if row.startswith("|"): row=row[1:]
    if row.endswith("|"): row=row[:-1]
    i=0
    while i<len(row):
        ch=row[i]
        if ch=="\\" and i+1<len(row) and row[i+1] in "|$": cur+=row[i:i+2]; i+=2; continue
        if ch=="$": inm=not inm
        if ch=="|" and not inm: cells.append(cur.strip()); cur=""
        else: cur+=ch
        i+=1
    cells.append(cur.strip())
    return [c for c in cells if c]
count={}
def title_of(h): return re.sub(r"^#+\s*","",h).strip()
changed=0
for f in order:
    if '/' not in f: continue
    c=chap[f.split('/')[0]]
    lines=open(f).read().split("\n")
    out=[];infence=False;heading="";i=0
    while i<len(lines):
        l=lines[i]
        if l.startswith("```"): infence=not infence
        if not infence and l.startswith("#"): heading=title_of(l)
        # skip if a caption already follows/precedes
        def nextnonblank(j):
            while j<len(lines) and not lines[j].strip(): j+=1
            return lines[j] if j<len(lines) else ""
        if not infence and l.strip()=='<div class="figure-row">':
            j=i
            while lines[j].strip()!='</div>': j+=1
            block=lines[i:j+1]
            out+=block
            if 'class="caption"' not in nextnonblank(j+1):
                alts=re.findall(r"!\[([^\]]*)\]","\n".join(block))
                k=count.setdefault((c,'F'),0)+1; count[(c,'F')]=k
                a=", ".join(alts[:-1])+" and "+alts[-1] if len(alts)>1 else alts[0]
                out+=["",f'<p class="caption" id="fig-{c}-{k}"><strong>Figure {c}.{k}.</strong> {a}.</p>']
            i=j+1; continue
        m=re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$",l)
        if not infence and m:
            if 'class="caption"' in nextnonblank(i+1): out.append(l); i+=1; continue
            k=count.setdefault((c,'F'),0)+1; count[(c,'F')]=k
            alt=m.group(1).strip()
            out+= [l,"",f'<p class="caption" id="fig-{c}-{k}"><strong>Figure {c}.{k}.</strong> {alt}.</p>']
            i+=1; continue
        m=re.match(r"^\{\{#include (\.\./generated/[^}]+\.md)\}\}\s*$",l)
        if not infence and m:
            gen=os.path.normpath(os.path.join(os.path.dirname(f),m.group(1)))
            first=open(gen).read().lstrip().split("\n")[0] if os.path.exists(gen) else ""
            prev=out[-1] if out else ""
            prevnb=next((x for x in reversed(out) if x.strip()),"")
            if first.startswith("|") and 'class="caption"' not in prevnb:
                k=count.setdefault((c,'T'),0)+1; count[(c,'T')]=k
                cells=splitcells(first)
                t="Columns: "+", ".join(cells[:4])+(", and more" if len(cells)>4 else "")
                out+=[f'<p class="caption" id="tab-{c}-{k}"><strong>Table {c}.{k}.</strong> {t}.</p>',""]
            out.append(l); i+=1; continue
        out.append(l); i+=1
    new="\n".join(out)
    if new!=open(f).read(): open(f,'w').write(new); changed+=1
print(changed,"files",{k:v for k,v in count.items() if k[1]=='F'} and sum(v for k,v in count.items() if k[1]=='F'),"figures",sum(v for k,v in count.items() if k[1]=='T'),"tables")

# Renumber every caption in reading order, so that inserting an entry, figure or table shifts the
# numbers after it. Captions keep their text; only "Figure C.k" / "Table C.k" and the ids change.
cap = re.compile(r'<p class="caption" id="(fig|tab)-\d+-\d+"><strong>(Figure|Table) \d+\.\d+\.</strong>')
seen = {}
renumbered = 0
for f in order:
    if '/' not in f: continue
    c = chap[f.split('/')[0]]
    s = open(f).read()
    def renum(m):
        kind = m.group(1)
        k = seen.get((c, kind), 0) + 1
        seen[(c, kind)] = k
        word = "Figure" if kind == "fig" else "Table"
        return f'<p class="caption" id="{kind}-{c}-{k}"><strong>{word} {c}.{k}.</strong>'
    t = cap.sub(renum, s)
    if t != s:
        open(f, 'w').write(t); renumbered += 1
print(renumbered, "files renumbered")
