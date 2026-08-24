import {
  ArrowUp,
  Clock3,
  FileText,
  Image as ImageIcon,
  LoaderCircle,
  Menu,
  Paperclip,
  Plus,
  Sparkles,
  X,
} from 'lucide-react'
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { Brand } from './components/Brand'
import { AuthScreen } from './components/AuthScreen'
import { EvidenceView } from './components/EvidenceView'
import { KnowledgeView } from './components/KnowledgeView'
import { MessageBubble } from './components/MessageBubble'
import { Sidebar } from './components/Sidebar'
import { api } from './lib/api'
import type {
  AppView,
  Attachment,
  AuthConfig,
  AuthUser,
  ConversationSummary,
  Health,
  KnowledgeStatus,
  Message,
  Source,
} from './types'

const welcome: Message = {
  id: 'welcome',
  role: 'assistant',
  content: '你好，我是 **BioCoder**。输入一个药物研发问题，我会制定检索计划、调用本地与公开数据源，并记录可评估的完整轨迹。',
}

const suggestions = [
  '分析 EGFR C797S 耐药机制及潜在应对策略',
  '检索 HER2 ADC 在乳腺癌中的近期临床试验',
  '比较奥希替尼的适应证与主要安全性风险',
]

const viewTitles: Record<AppView, { eyebrow: string; title: string }> = {
  chat: { eyebrow: 'AGENT CONSOLE', title: '药物研发智能分析' },
  knowledge: { eyebrow: 'LOCAL KNOWLEDGE', title: '医药知识检索' },
  evidence: { eyebrow: 'SOURCE REVIEW', title: '证据审查' },
}

const fallbackAttachmentFormats = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.pdf', '.docx', '.txt', '.md', '.json']

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

export default function App() {
  const [user, setUser] = useState<AuthUser>()
  const [authConfig, setAuthConfig] = useState<AuthConfig>()
  const [authLoading, setAuthLoading] = useState(true)
  const [messages, setMessages] = useState<Message[]>([welcome])
  const [input, setInput] = useState('')
  const [threadId, setThreadId] = useState<string>()
  const [health, setHealth] = useState<Health>()
  const [knowledge, setKnowledge] = useState<KnowledgeStatus>()
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeView, setActiveView] = useState<AppView>('chat')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [pendingFiles, setPendingFiles] = useState<File[]>([])
  const [busyStage, setBusyStage] = useState<'uploading' | 'thinking'>('thinking')
  const bottom = useRef<HTMLDivElement>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const refreshStatus = () => {
    api.health().then(setHealth).catch(() => undefined)
    api.knowledge().then(setKnowledge).catch(() => undefined)
  }
  const refreshConversations = () => api.conversations().then(setConversations).catch(() => undefined)

  useEffect(() => {
    api.health().then(setHealth).catch(() => undefined)
    api.authConfig().then(setAuthConfig).catch(() => undefined)
    api.me()
      .then((authenticatedUser) => {
        setUser(authenticatedUser)
        refreshStatus()
        refreshConversations()
      })
      .catch(() => undefined)
      .finally(() => setAuthLoading(false))
  }, [])
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  const evidenceSources = useMemo(() => {
    const seen = new Set<string>()
    return messages.flatMap((message) => message.sources || []).filter((source) => {
      const key = `${source.title}|${source.url || ''}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
  }, [messages])

  async function send(text = input) {
    const files = [...pendingFiles]
    const question = text.trim() || (files.length ? '请分析这些附件并总结关键发现。' : '')
    if ((!question && files.length === 0) || busy) return
    setError('')
    setBusy(true)
    setBusyStage(files.length ? 'uploading' : 'thinking')
    setActiveView('chat')
    try {
      const attachments: Attachment[] = []
      for (const file of files) attachments.push(await api.uploadAttachment(file))
      setInput('')
      setPendingFiles([])
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'user', content: question, attachments },
      ])
      setBusyStage('thinking')
      const result = await api.chat(question, threadId, attachments.map((item) => item.id))
      setThreadId(result.thread_id)
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.answer,
          task_id: result.task_id,
          plan: result.plan,
          sources: result.sources,
          tools: result.tools_used,
        },
      ])
      refreshStatus()
      refreshConversations()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '请求失败，请检查后端服务。')
    } finally {
      setBusy(false)
      setBusyStage('thinking')
    }
  }

  function addAttachments(files: FileList | null) {
    if (!files?.length) return
    const formats = health?.attachment_formats || fallbackAttachmentFormats
    const maximumFiles = health?.attachment_max_files || 4
    const maximumBytes = health?.attachment_max_file_bytes || 15 * 1024 * 1024
    const next = [...pendingFiles]
    for (const file of Array.from(files)) {
      const dot = file.name.lastIndexOf('.')
      const suffix = dot >= 0 ? file.name.slice(dot).toLowerCase() : ''
      if (!formats.includes(suffix)) {
        setError(`不支持“${file.name}”的格式。支持：${formats.join('、')}`)
        continue
      }
      if (file.size > maximumBytes) {
        setError(`“${file.name}”超过单文件 ${formatBytes(maximumBytes)} 限制。`)
        continue
      }
      if (health && !health.vision_input_enabled && file.type.startsWith('image/')) {
        setError('当前模型未启用视觉输入，不能添加图片。')
        continue
      }
      if (next.some((item) => item.name === file.name && item.size === file.size)) continue
      if (next.length >= maximumFiles) {
        setError(`每条消息最多添加 ${maximumFiles} 个附件。`)
        break
      }
      next.push(file)
    }
    setPendingFiles(next)
    if (fileInput.current) fileInput.current.value = ''
  }

  function submit(event: FormEvent) {
    event.preventDefault()
    void send()
  }

  function newConversation() {
    setMessages([welcome])
    setThreadId(undefined)
    setInput('')
    setPendingFiles([])
    setError('')
    setActiveView('chat')
    setSidebarOpen(false)
  }

  async function loadConversation(id: string) {
    if (busy) return
    setError('')
    try {
      const conversation = await api.conversation(id)
      setMessages(conversation.messages.map((message) => ({ ...message, id: `stored-${message.id}` })))
      setPendingFiles([])
      setThreadId(conversation.thread_id)
      setActiveView('chat')
      setSidebarOpen(false)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '历史会话加载失败')
    }
  }

  async function deleteConversation(id: string) {
    try {
      await api.deleteConversation(id)
      if (id === threadId) newConversation()
      refreshConversations()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '删除失败')
    }
  }

  async function upload(file: File) {
    setBusy(true)
    setError('')
    try { setKnowledge(await api.upload(file)) }
    catch (caught) { setError(caught instanceof Error ? caught.message : '上传失败') }
    finally { setBusy(false) }
  }

  async function reindex() {
    setBusy(true)
    setError('')
    try { setKnowledge(await api.reindex()) }
    catch (caught) { setError(caught instanceof Error ? caught.message : '索引失败') }
    finally { setBusy(false) }
  }

  function changeView(view: AppView) {
    setActiveView(view)
    setSidebarOpen(false)
  }

  function authenticated(authenticatedUser: AuthUser) {
    setUser(authenticatedUser)
    newConversation()
    refreshStatus()
    refreshConversations()
  }

  async function logout() {
    await api.logout()
    setUser(undefined)
    setConversations([])
    setKnowledge(undefined)
    newConversation()
  }

  if (authLoading) {
    return <div className="app-loading"><Brand /><LoaderCircle size={21} /></div>
  }

  if (!user) {
    return (
      <AuthScreen
        config={authConfig}
        onLogin={async (email, password) => {
          const authenticatedUser = await api.login(email, password)
          authenticated(authenticatedUser)
          return authenticatedUser
        }}
        onRegister={async (displayName, email, password, inviteCode) => {
          const authenticatedUser = await api.register(displayName, email, password, inviteCode)
          authenticated(authenticatedUser)
          return authenticatedUser
        }}
      />
    )
  }

  return (
    <div className="app-shell">
      <div className={`mobile-backdrop ${sidebarOpen ? 'show' : ''}`} onClick={() => setSidebarOpen(false)} />
      <div className={`sidebar-wrap ${sidebarOpen ? 'open' : ''}`}>
        <button className="sidebar-close" onClick={() => setSidebarOpen(false)}><X size={19} /></button>
        <Sidebar
          health={health}
          knowledge={knowledge}
          conversations={conversations}
          activeThread={threadId}
          activeView={activeView}
          onViewChange={changeView}
          onConversation={(id) => void loadConversation(id)}
          onDeleteConversation={(id) => void deleteConversation(id)}
          onNew={newConversation}
          user={user}
          onLogout={() => void logout()}
        />
      </div>

      <main className="workspace">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)}><Menu size={19} /></button>
          <div className="mobile-brand"><Brand compact /></div>
          <div className="page-identity"><span>{viewTitles[activeView].eyebrow}</span><h1>{viewTitles[activeView].title}</h1></div>
          <div className="topbar-actions">
            {threadId && <div className="thread-pill"><Clock3 size={13} /> 历史已保存</div>}
            <button className="new-chat" onClick={newConversation}><Plus size={15} /> 新对话</button>
          </div>
        </header>

        {activeView === 'chat' && (
          <>
            <div className="chat-scroll">
              <div className="chat-column">
                {messages.map((message) => <MessageBubble message={message} key={message.id} />)}

                {messages.length === 1 && (
                  <div className="suggestions">
                    <div><Sparkles size={14} /> RESEARCH STARTERS</div>
                    {suggestions.map((item, index) => <button key={item} onClick={() => void send(item)}><span>{String(index + 1).padStart(2, '0')}</span><b>{item}</b><i>↗</i></button>)}
                  </div>
                )}

                {busy && (
                  <div className="thinking">
                    <LoaderCircle size={18} />
                    <div>
                      <strong>{busyStage === 'uploading' ? '正在处理附件' : 'Agent 正在工作'}</strong>
                      <span>{busyStage === 'uploading' ? '校验 → 解析 → 视觉预处理' : '规划 → 检索 → 交叉验证 → 汇总'}</span>
                    </div>
                  </div>
                )}
                {error && <div className="error-banner"><strong>暂时无法完成请求</strong><span>{error}</span></div>}
                <div ref={bottom} />
              </div>
            </div>

            <div className="composer-wrap">
              <form className="composer" onSubmit={submit}>
                {pendingFiles.length > 0 && (
                  <div className="composer-attachments">
                    {pendingFiles.map((file) => (
                      <div className="attachment-chip" key={`${file.name}-${file.size}`}>
                        {file.type.startsWith('image/') ? <ImageIcon size={14} /> : <FileText size={14} />}
                        <span><b>{file.name}</b><small>{formatBytes(file.size)}</small></span>
                        <button
                          type="button"
                          aria-label={`移除 ${file.name}`}
                          onClick={() => setPendingFiles((current) => current.filter((item) => item !== file))}
                        >
                          <X size={13} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <input
                  ref={fileInput}
                  className="attachment-input"
                  type="file"
                  multiple
                  accept={(health?.attachment_formats || fallbackAttachmentFormats).join(',')}
                  onChange={(event) => addAttachments(event.currentTarget.files)}
                />
                <button
                  type="button"
                  className="attachment-button"
                  aria-label="添加附件"
                  title="添加图片、PDF、Word 或文本附件"
                  disabled={busy || health?.attachments_enabled === false}
                  onClick={() => fileInput.current?.click()}
                >
                  <Paperclip size={18} />
                </button>
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void send() }
                  }}
                  placeholder="描述一个药物、靶点、疾病或临床研究问题…"
                  rows={1}
                />
                <button
                  className="send-button"
                  aria-label="发送"
                  disabled={busy || (!input.trim() && pendingFiles.length === 0)}
                >
                  <ArrowUp size={18} />
                </button>
              </form>
              <div className="composer-note">支持图片、PDF、DOCX 与文本 · 单条最多 {health?.attachment_max_files || 4} 个</div>
            </div>
          </>
        )}

        {activeView === 'knowledge' && (
          <div className="view-scroll">
            <KnowledgeView
              knowledge={knowledge}
              busy={busy}
              onUpload={upload}
              onReindex={reindex}
              onKnowledgeChange={setKnowledge}
            />
          </div>
        )}
        {activeView === 'evidence' && (
          <div className="view-scroll"><EvidenceView sources={evidenceSources as Source[]} /></div>
        )}
      </main>
    </div>
  )
}
