import "./App.css";
import { Routes, Route, useNavigate } from "react-router-dom";

//screens import
import Landing from "./screens/landing/landing";
import About from "./screens/about/about";
import Products from "./screens/products/products";

//navbar
import Nav from "./Navbar/nav";

//footer
import Footer from "./footer/footer";

function App() {
  let navigate = useNavigate();

  return (
    <div className="App">
      <Nav navigate={navigate} />
      <Routes>
        <Route path="/" element={<Landing navigate={navigate} />} />
        <Route path="/about" element={<About navigate={navigate} />} />
        <Route path="/products" element={<Products navigate={navigate} />} />
      </Routes>
      <Footer />
    </div>
  );
}

export default App;
