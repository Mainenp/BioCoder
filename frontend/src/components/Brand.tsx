export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand ${compact ? 'compact' : ''}`} aria-label="BioCoder">
      <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
      <span className="brand-name">BIO<span>/</span>CODER</span>
      {!compact && <small>PHARMA RESEARCH OS</small>}
    </div>
  )
}
