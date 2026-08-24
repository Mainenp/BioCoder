import { ArrowRight, AtSign, KeyRound, LockKeyhole, Mail, ShieldCheck, UserRound } from 'lucide-react'
import { FormEvent, useState } from 'react'
import type { AuthConfig, AuthUser } from '../types'
import { Brand } from './Brand'

type Props = {
  config?: AuthConfig
  onLogin: (email: string, password: string) => Promise<AuthUser>
  onRegister: (
    displayName: string,
    email: string,
    password: string,
    inviteCode: string,
  ) => Promise<AuthUser>
}

export function AuthScreen({ config, onLogin, onRegister }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [displayName, setDisplayName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [inviteCode, setInviteCode] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const requestSubject = encodeURIComponent('申请 BioCoder 邀请码')
  const requestBody = encodeURIComponent('您好，我希望申请 BioCoder 使用邀请码。\n\n申请人：\n用途：\n')
  const mailLink = config?.admin_email
    ? `mailto:${config.admin_email}?subject=${requestSubject}&body=${requestBody}`
    : undefined

  async function submit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      if (mode === 'login') await onLogin(email.trim(), password)
      else await onRegister(displayName.trim(), email.trim(), password, inviteCode.trim())
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '暂时无法完成认证，请稍后重试。')
    } finally {
      setBusy(false)
    }
  }

  function switchMode(next: 'login' | 'register') {
    setMode(next)
    setError('')
  }

  return (
    <main className="auth-shell">
      <section className="auth-story">
        <div className="auth-story-top"><Brand /><span>PRIVATE RESEARCH WORKSPACE</span></div>
        <div className="auth-orbit" aria-hidden="true">
          <i /><i /><i /><b>BC</b>
        </div>
        <div className="auth-story-copy">
          <span className="auth-kicker">BIOCODER / RESEARCH OS</span>
          <h1>让每一次研究<br />都只属于你。</h1>
          <p>独立账号、隔离会话与可追溯证据，在一个安静、专注的药物研发工作台中持续积累。</p>
        </div>
        <div className="auth-trust-row">
          <span><ShieldCheck size={15} /> 会话按账号隔离</span>
          <span><LockKeyhole size={15} /> 安全会话 Cookie</span>
        </div>
      </section>

      <section className="auth-panel">
        <div className="auth-card">
          <div className="auth-card-head">
            <span>{mode === 'login' ? 'WELCOME BACK' : 'INVITE ONLY'}</span>
            <h2>{mode === 'login' ? '登录研究空间' : '创建你的账号'}</h2>
            <p>{mode === 'login' ? '继续查看你的对话、证据与研究进度。' : '注册需要管理员提供的邀请码。'}</p>
          </div>

          <div className="auth-tabs" role="tablist" aria-label="账号操作">
            <button className={mode === 'login' ? 'active' : ''} onClick={() => switchMode('login')} type="button">登录</button>
            <button className={mode === 'register' ? 'active' : ''} onClick={() => switchMode('register')} type="button" disabled={config?.registration_enabled === false}>注册</button>
          </div>

          <form className="auth-form" onSubmit={submit}>
            {mode === 'register' && (
              <label>
                <span>昵称</span>
                <div><UserRound size={17} /><input required minLength={2} maxLength={40} autoComplete="name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="你的称呼" /></div>
              </label>
            )}
            <label>
              <span>邮箱</span>
              <div><AtSign size={17} /><input required type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="name@example.com" /></div>
            </label>
            <label>
              <span>密码</span>
              <div><LockKeyhole size={17} /><input required minLength={8} maxLength={128} type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} value={password} onChange={(event) => setPassword(event.target.value)} placeholder="至少 8 个字符" /></div>
            </label>
            {mode === 'register' && (
              <label>
                <span>邀请码</span>
                <div><KeyRound size={17} /><input required autoComplete="off" value={inviteCode} onChange={(event) => setInviteCode(event.target.value)} placeholder="输入管理员提供的邀请码" /></div>
              </label>
            )}

            {error && <div className="auth-error">{error}</div>}
            <button className="auth-submit" disabled={busy}>
              <span>{busy ? '请稍候…' : mode === 'login' ? '进入工作台' : '创建账号'}</span>
              {!busy && <ArrowRight size={17} />}
            </button>
          </form>

          {mode === 'register' && (
            <div className="invite-help">
              <Mail size={17} />
              <div><strong>还没有邀请码？</strong><span>请发送申请邮件给管理员，说明你的姓名与使用用途。</span></div>
              {mailLink && <a href={mailLink}>发送邮件</a>}
            </div>
          )}
        </div>
        <div className="auth-footnote">RESEARCH USE ONLY · NOT FOR CLINICAL DECISIONS</div>
      </section>
    </main>
  )
}
