"""从 artifact_tool 创建的数字模板生成内存受限的XLSX。

仅支持本项目自带的简单模板。使用Python标准库的ZIP/XML序列化器填充
sheetData；复现百万单元输出不依赖Excel或第三方表格引擎。
元数据、表头、样式和列宽均保留，写入过程采用原子替换。
"""
from __future__ import annotations
import csv, gzip, math, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from itertools import islice

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ET.register_namespace('x', NS)

def column_name(n):
    text='';n+=1
    while n:
        n,rem=divmod(n-1,26);text=chr(65+rem)+text
    return text

def _parts(xml,row_count,ncols):
    root=ET.fromstring(xml);sd=root.find(f'{{{NS}}}sheetData')
    rows=list(sd)
    if len(rows)!=2:
        raise ValueError('Use the shipped two-row templates, not an arbitrary workbook.')
    styles={c.attrib['r']:c.attrib.get('s','0') for c in rows[1]}
    st=styles.get('A2','0');sv=styles.get('B2','0')
    for row in rows[1:]:sd.remove(row)
    sd.append(ET.Element('STREAM_PLACEHOLDER'))
    dim=root.find(f'{{{NS}}}dimension')
    if dim is None:
        dim=ET.Element(f'{{{NS}}}dimension')
        root.insert(1 if root.find(f'{{{NS}}}sheetPr') is not None else 0,dim)
    dim.set('ref',f'A1:{column_name(ncols-1)}{row_count+1}')
    views=root.find(f'{{{NS}}}sheetViews')
    if views is None:
        views=ET.Element(f'{{{NS}}}sheetViews');root.insert(list(root).index(dim)+1,views)
    views.clear()
    view=ET.SubElement(views,f'{{{NS}}}sheetView',{'workbookViewId':'0'})
    ET.SubElement(view,f'{{{NS}}}pane',{'xSplit':'1','ySplit':'1','topLeftCell':'B2',
                                      'activePane':'bottomRight','state':'frozen'})
    ET.SubElement(view,f'{{{NS}}}selection',{'pane':'bottomRight','activeCell':'B2','sqref':'B2'})
    encoded=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    pre,post=encoded.split(b'<STREAM_PLACEHOLDER />')
    return pre,post,st,sv

def stream_xlsx(template, csv_by_sheet, target, expected_rows, allow_blank=False):
    """csv_by_sheet 将实际工作表名映射到CSV/CSV.GZ文件（包含表头）。"""
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if not 0<expected_rows<=1048575:
        raise ValueError('XLSX row limit exceeded. Full CSV files remain available; use --no-excel.')
    tmp=target.with_suffix('.xlsx.partial')
    try:
        with zipfile.ZipFile(template) as seed, zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,
                                                               compresslevel=5,allowZip64=True) as out:
            wb=ET.fromstring(seed.read('xl/workbook.xml'))
            rels=ET.fromstring(seed.read('xl/_rels/workbook.xml.rels'))
            links={e.attrib['Id']:e.attrib['Target'] for e in rels}
            mapping={}
            for s in wb.find(f'{{{NS}}}sheets'):
                name=s.attrib['name']
                if name in csv_by_sheet:
                    loc=links[s.attrib[f'{{{REL}}}id']]
                    loc=loc.lstrip('/') if loc.startswith('/') else 'xl/'+loc
                    mapping[loc]=Path(csv_by_sheet[name])
            if len(mapping)!=len(csv_by_sheet):raise ValueError('Template worksheet names do not match CSV mapping.')
            for item in seed.infolist():
                if item.filename not in mapping:
                    out.writestr(item.filename,seed.read(item.filename));continue
                p=mapping[item.filename]
                opener=gzip.open if p.suffix=='.gz' else open
                with opener(p,'rt',encoding='utf-8-sig',newline='') as f, out.open(item.filename,'w',force_zip64=True) as stream:
                    reader=csv.reader(f);headers=next(reader);nc=len(headers)
                    pre,post,st,sv=_parts(seed.read(item.filename),expected_rows,nc)
                    stream.write(pre);nr=1;prev=-math.inf
                    col=[column_name(j) for j in range(nc)]
                    while True:
                        batch=list(islice(reader,1500))
                        if not batch:break
                        pieces=[]
                        for vals in batch:
                            if len(vals)!=nc:raise ValueError('CSV column count changed.')
                            ts=float(vals[0])
                            if ts<=prev:raise ValueError('Time values must be strictly increasing.')
                            prev=ts;nr+=1;pieces.append(f'<x:row r="{nr}">')
                            for j,v in enumerate(vals):
                                if v=='':
                                    if not allow_blank:raise ValueError('Unexpected blank numeric value.')
                                    continue
                                value=float(v)
                                if not math.isfinite(value):raise ValueError('Nonfinite XLSX value.')
                                style=st if j==0 else sv
                                pieces.append(f'<x:c r="{col[j]}{nr}" s="{style}" t="n"><x:v>{value:.15g}</x:v></x:c>')
                            pieces.append('</x:row>')
                        stream.write(''.join(pieces).encode('utf-8'))
                    if nr-1!=expected_rows:raise ValueError(f'Expected {expected_rows} rows, found {nr-1}.')
                    stream.write(post)
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True);raise
    return target
