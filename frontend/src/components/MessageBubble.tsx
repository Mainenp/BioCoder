import { Bot, CheckCircle2, FileText, Image as ImageIcon, UserRound, Wrench } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../types'
import { AnswerFeedback } from './AnswerFeedback'
import { SourceCard } from './SourceCard'

function toolLabel(value: string) {
  if (value === 'read_attachment') return '附件解析'
  return value.replace('search_', '')
}

export function MessageBubble({ message }: { message: Message }) {
  const assistant = message.role === 'assistant'
  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">{assistant ? <Bot size={19} /> : <UserRound size={18} />}</div>
      <div className="message-main">
        <div className="message-meta">{assistant ? 'BioCoder' : '你'}</div>

        {message.plan && message.plan.length > 0 && (
          <section className="plan-card">
            <div className="plan-title"><CheckCircle2 size={16} /> 本轮分析计划</div>
            <ol>{message.plan.map((step) => <li key={step}>{step}</li>)}</ol>
          </section>
        )}

        {message.attachments && message.attachments.length > 0 && (
          <div className="message-attachments">
            {message.attachments.map((attachment) => (
              <div className="message-attachment" key={attachment.id}>
                {attachment.kind === 'image' ? <ImageIcon size={15} /> : <FileText size={15} />}
                <span><b>{attachment.name}</b><small>{attachment.kind.toUpperCase()}</small></span>
              </div>
            ))}
          </div>
        )}

        <div className="message-content">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({ children }) => (
                <div
                  className="markdown-table-wrap"
                  role="region"
                  aria-label="数据表格，可横向滚动"
                  tabIndex={0}
                >
                  <table>{children}</table>
                </div>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {message.tools && message.tools.length > 0 && (
          <div className="tools-row"><Wrench size={13} /> 已调用 {message.tools.map(toolLabel).join(' · ')}</div>
        )}

        {message.sources && message.sources.length > 0 && (
          <section className="sources">
            <div className="sources-title">证据来源 <span>{message.sources.length}</span></div>
            <div className="source-grid">
              {message.sources.map((source, index) => (
                <SourceCard source={source} index={index} key={`${source.title}-${index}`} />
              ))}
            </div>
          </section>
        )}

        {assistant && message.task_id && <AnswerFeedback taskId={message.task_id} />}
      </div>
    </article>
  )
}
