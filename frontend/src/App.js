import './App.css';
import { TextField, Box } from '@mui/material';

function App() {
  return (
    <div className="App">
      <p>
        <code>Title</code>
      </p>
      <header className="App-header">
      <Box
      component="form"
      sx={{
        '& > :not(style)': { m: 1, width: '25ch' },
      }}
      noValidate
      autoComplete="off"
    >
        <TextField id="outlined-basic" label="Input" variant="outlined" color="primary" />
        <TextField id="outlined-basic" label="Output" variant="outlined" color="primary" />
        </Box>
      </header>
    </div>
  );
}

export default App;
