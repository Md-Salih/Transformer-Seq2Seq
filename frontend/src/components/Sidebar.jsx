import { motion } from 'framer-motion'
import './Sidebar.css'

const EXAMPLES = [
  {
    title: 'AI Technology',
    text: 'Artificial Intelligence has evolved from a theoretical concept to a transformative technology that is reshaping industries and society. Modern AI systems, powered by deep learning and neural networks, can now perform tasks that were once thought to be exclusively human domains, including image recognition, natural language processing, and complex decision-making.'
  },
  {
    title: 'Climate Change',
    text: 'Climate change represents one of the most significant challenges facing humanity in the 21st century. Rising global temperatures are causing widespread environmental disruptions, including melting polar ice caps, rising sea levels, and increasingly frequent extreme weather events.'
  },
  {
    title: 'Space Exploration',
    text: 'Space exploration has entered a new era characterized by international collaboration, private sector innovation, and ambitious goals. Recent missions have successfully landed rovers on Mars, collected samples from asteroids, and deployed powerful telescopes that peer into the early universe.'
  }
]

const Sidebar = ({ onNewChat, chatHistory = [], onSelectChat, onExampleClick }) => {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <button className="new-chat-btn" onClick={onNewChat}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M12 5v14M5 12h14" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          New chat
        </button>
      </div>
      
      <div className="sidebar-content">
        <div className="examples-section">
          <h3 className="section-title">Examples</h3>
          <div className="example-list">
            {EXAMPLES.map((example, index) => (
              <button 
                key={index} 
                className="example-item"
                onClick={() => onExampleClick(example.text)}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" strokeWidth="2"/>
                </svg>
                <span className="example-title">{example.title}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="chat-list">
          {chatHistory.map((chat, index) => (
            <button 
              key={index} 
              className="chat-item"
              onClick={() => onSelectChat(index)}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" strokeWidth="2"/>
              </svg>
              <span className="chat-title">{chat.title}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="sidebar-footer">
        <div className="user-info">
          <div className="user-avatar">AI</div>
          <span className="user-name">Transformer Seq2Seq</span>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
