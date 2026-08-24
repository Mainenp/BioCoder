import { Circle, FlaskConical, LogOut, MessageSquareText, Plus, Search, ShieldCheck, Trash2 } from 'lucide-react'
import type { AppView, AuthUser, ConversationSummary, Health, KnowledgeStatus } from '../types'
import { Brand } from './Brand'

type Props = {
  health?: Health
  knowledge?: KnowledgeStatus
  conversations: ConversationSummary[]
  activeThread?: string
  activeView: AppView
  onViewChange: (view: AppView) => void
  onConversation: (threadId: string) => void
  onDeleteConversation: (threadId: string) => void
  onNew: () => void
  user: AuthUser
  onLogout: () => void
}

const nav = [
  { id: 'chat' as const, label: '研究工作台', hint: 'Agent Console', icon: MessageSquareText },
  { id: 'knowledge' as const, label: '知识检索', hint: 'Local RAG', icon: Search },
  { id: 'evidence' as const, label: '证据审查', hint: 'Source Review', icon: ShieldCheck },
]

function relativeTime(value: string) {
  const date = new Date(value)
  const diff = Date.now() - date.getTime()
  if (diff < 60_000) return '刚刚'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`
  return `${date.getMonth() + 1}/${date.getDate()}`
}

export function Sidebar({
  health,
  knowledge,
  conversations,
  activeThread,
  activeView,
  onViewChange,
  onConversation,
  onDeleteConversation,
  onNew,
  user,
  onLogout,
}: Props) {
  return (
    <aside className="sidebar">
      <Brand />

      <button className="create-chat" onClick={onNew}><Plus size={16} /> 新建研究</button>

      <nav className="nav-list" aria-label="主导航">
        {nav.map(({ id, label, hint, icon: Icon }) => (
          <button className={`nav-item ${activeView === id ? 'active' : ''}`} key={id} onClick={() => onViewChange(id)}>
            <Icon size={17} /><span><b>{label}</b><small>{hint}</small></span><i>↗</i>
          </button>
        ))}
      </nav>

      <section className="history-section">
        <div className="sidebar-label"><span>RECENT THREADS</span><b>{conversations.length}</b></div>
        <div className="history-items">
          {conversations.map((conversation) => (
            <div
              className={`history-entry ${activeThread === conversation.thread_id ? 'active' : ''}`}
              key={conversation.thread_id}
            >
              <button className="history-select" onClick={() => onConversation(conversation.thread_id)}>
                <MessageSquareText size={14} />
                <span><b>{conversation.title}</b><small>{relativeTime(conversation.updated_at)} · {conversation.message_count} 条消息</small></span>
              </button>
              <button
                className="history-delete"
                aria-label="删除会话"
                onClick={() => onDeleteConversation(conversation.thread_id)}
              ><Trash2 size={13} /></button>
            </div>
          ))}
          {!conversations.length && <div className="history-empty"><FlaskConical size={19} /><span>完成首次分析后<br />会话会保存在这里</span></div>}
        </div>
      </section>

      <div className="runtime-card">
        <div><Circle size={8} fill={health?.status === 'ok' ? 'currentColor' : 'none'} /><span>LOCAL RUNTIME</span><b>{health?.status === 'ok' ? 'ONLINE' : 'OFFLINE'}</b></div>
        <div><span>LLM</span><b>{health?.llm_configured ? 'READY' : 'NO KEY'}</b></div>
        <div><span>KNOWLEDGE</span><b>{knowledge?.chunks ?? 0} CHUNKS</b></div>
      </div>

      <div className="account-card">
        <div className="account-avatar">{user.display_name.slice(0, 1).toUpperCase()}</div>
        <span><b>{user.display_name}</b><small>{user.email}</small></span>
        <button onClick={onLogout} aria-label="退出登录" title="退出登录"><LogOut size={14} /></button>
      </div>

      <div className="disclaimer">BIOAGENT / RESEARCH USE ONLY<br />NOT FOR CLINICAL DECISIONS</div>
    </aside>
  )
}
