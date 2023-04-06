import React from 'react'
import "./nav.css"

function Nav({navigate}) {
  return (
    <div className='navContainer'>
        <div className='navLogo'>
            <h1
                onClick={() => navigate('/')} 
                style={{
                    cursor: 'pointer'
                }}
            >WatchGuard AI</h1>
        </div>
        <div className='navLinks'>
            <ul>
                <li
                    onClick={() => navigate('/products')}
                    style={{
                      cursor: 'pointer',
                    }}
                >Products</li>
                <li
                    onClick={() => navigate('/about')}
                    style={{
                      cursor: 'pointer',
                    }}
                >About</li>
            </ul>
        </div>
    </div>
  )
}

export default Nav