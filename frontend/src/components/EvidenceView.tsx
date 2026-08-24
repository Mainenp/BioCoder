import { ExternalLink, FileCheck2, ShieldCheck } from 'lucide-react'
import type { Source } from '../types'

export function EvidenceView({ sources }: { sources: Source[] }) {
  const grouped = sources.reduce<Record<string, number>>((counts, source) => {
    counts[source.source_type] = (counts[source.source_type] || 0) + 1
    return counts
  }, {})

  return (
    <section className="feature-view">
      <div className="view-heading">
        <div><span className="section-kicker">EVIDENCE REVIEW</span><h2>审查当前会话的证据</h2></div>
        <div className="view-metric"><strong>{sources.length}</strong><span>TRACEABLE SOURCES</span></div>
      </div>

      <div className="evidence-stats">
        <article><ShieldCheck size={20} /><div><strong>{sources.filter((source) => source.url).length}</strong><span>可访问原文</span></div></article>
        <article><FileCheck2 size={20} /><div><strong>{Object.keys(grouped).length}</strong><span>来源类型</span></div></article>
        {Object.entries(grouped).map(([type, count]) => <article key={type}><div><strong>{count}</strong><span>{type.replaceAll('_', ' ')}</span></div></article>)}
      </div>

      {sources.length > 0 ? (
        <div className="evidence-table">
          <div className="evidence-row evidence-head"><span>编号</span><span>来源与摘要</span><span>类型</span><span>原文</span></div>
          {sources.map((source, index) => (
            <div className="evidence-row" key={`${source.title}-${index}`}>
              <span>{String(index + 1).padStart(2, '0')}</span>
              <div><strong>{source.title}</strong><p>{source.snippet || '暂无摘要'}</p></div>
              <span>{source.source_type.replaceAll('_', ' ')}</span>
              <span>{source.url ? <a href={source.url} target="_blank" rel="noreferrer"><ExternalLink size={15} /></a> : 'LOCAL'}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="feature-empty evidence-empty"><ShieldCheck size={30} /><strong>当前会话还没有证据</strong><p>完成一次带工具检索的 Agent 分析后，来源会自动汇总到这里。</p></div>
      )}
    </section>
  )
}
