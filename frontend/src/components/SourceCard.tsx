import { ExternalLink, FileText, FlaskConical, Image as ImageIcon, Pill, ScrollText } from 'lucide-react'
import type { Source } from '../types'

const icons = {
  pubmed: ScrollText,
  openfda: Pill,
  clinical_trial: FlaskConical,
  local_knowledge: FileText,
  attachment_image: ImageIcon,
  attachment_pdf: FileText,
  attachment_word: FileText,
  attachment_text: FileText,
}

export function SourceCard({ source, index }: { source: Source; index: number }) {
  const Icon = icons[source.source_type as keyof typeof icons] || FileText
  const body = (
    <>
      <div className="source-index">{index + 1}</div>
      <div className="source-icon"><Icon size={17} /></div>
      <div className="source-copy">
        <strong>{source.title}</strong>
        <span>{source.source_type.replace('_', ' ')}</span>
        {source.snippet && <p>{source.snippet}</p>}
      </div>
      {source.url && <ExternalLink size={15} className="source-link-icon" />}
    </>
  )
  return source.url ? (
    <a className="source-card" href={source.url} target="_blank" rel="noreferrer">{body}</a>
  ) : (
    <div className="source-card">{body}</div>
  )
}
