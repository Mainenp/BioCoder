import { Database, FileUp, RefreshCw, Search } from 'lucide-react'
import { FormEvent, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { KnowledgeStatus, Source } from '../types'
import { SourceCard } from './SourceCard'

type Props = {
  knowledge?: KnowledgeStatus
  busy: boolean
  onUpload: (file: File) => void
  onReindex: () => void
  onKnowledgeChange: (status: KnowledgeStatus) => void
}

export function KnowledgeView({ knowledge, busy, onUpload, onReindex, onKnowledgeChange }: Props) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<Source[]>([])
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)

  async function search(event: FormEvent) {
    event.preventDefault()
    if (!query.trim()) return
    setSearching(true)
    setError('')
    try {
      setResults(await api.searchKnowledge(query.trim()))
      onKnowledgeChange(await api.knowledge())
    }
    catch (caught) { setError(caught instanceof Error ? caught.message : '检索失败') }
    finally { setSearching(false) }
  }

  return (
    <section className="feature-view">
      <div className="view-heading">
        <div><span className="section-kicker">KNOWLEDGE RETRIEVAL</span><h2>检索你的医药知识库</h2></div>
        <div className="view-metric"><strong>{knowledge?.chunks ?? 0}</strong><span>INDEXED CHUNKS</span></div>
      </div>

      <form className="knowledge-search" onSubmit={search}>
        <Search size={19} />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入药物、靶点、机制或试验关键词…" />
        <button disabled={searching || !query.trim()}>{searching ? '检索中' : '检索'}</button>
      </form>

      <div className="knowledge-layout">
        <div className="knowledge-files">
          <div className="panel-label">SOURCE FILES</div>
          <div className="knowledge-summary"><Database size={22} /><div><strong>{knowledge?.documents ?? 0} 个文档</strong><span>{knowledge?.ready ? '向量索引已就绪' : '等待首次索引'}</span></div></div>
          <div className="document-list">
            {knowledge?.files.map((file, index) => <div key={file}><span>{String(index + 1).padStart(2, '0')}</span>{file}</div>)}
            {!knowledge?.files.length && <p>还没有可检索的文档。</p>}
          </div>
          <input ref={fileInput} hidden type="file" accept=".md,.txt,.pdf,.json" onChange={(event) => event.target.files?.[0] && onUpload(event.target.files[0])} />
          <div className="knowledge-actions">
            <button disabled={busy} onClick={() => fileInput.current?.click()}><FileUp size={15} /> 上传资料</button>
            <button disabled={busy} onClick={onReindex}><RefreshCw size={15} /> 重建索引</button>
          </div>
        </div>

        <div className="retrieval-results">
          <div className="panel-label">RETRIEVAL RESULTS · {results.length}</div>
          {error && <div className="inline-error">{error}</div>}
          {results.length > 0 ? (
            <div className="retrieval-list">{results.map((source, index) => <SourceCard key={`${source.title}-${index}`} source={source} index={index} />)}</div>
          ) : (
            <div className="feature-empty"><Search size={28} /><strong>等待检索</strong><p>结果将按相关度展示，并保留原始文件与片段信息。</p></div>
          )}
        </div>
      </div>
    </section>
  )
}
