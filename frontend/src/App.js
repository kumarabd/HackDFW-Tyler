import './App.css';
import { useState } from 'react';
import { TextField, Box } from '@mui/material';

function App() {
    const inputProps = {
        step: 300,
        defaultInputValue: "",
    }

    const [data, setData] = useState('Empty');

    const handleTextInputChange = async e => {
        const response = await fetch(`http://127.0.0.1:8081/api/input`, {
            method: 'POST',
            crossDomain: true,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: e.target.value })
        })
        const result = await response.json()
        console.log(result)
        setData(result.data)

    }

    return ( <
        div className = "App" >
        <
        p >
        <
        code > Bias Removal < /code> <
        /p> <
        header className = "App-header" >
        <
        Box component = "form"
        sx = {
            {
                '& > :not(style)': { m: 1, width: '25ch' },
            }
        }
        noValidate autoComplete = "off" >
        <
        TextField id = "input"
        label = "Input"
        variant = "outlined"
        color = "primary"
        onChange = { handleTextInputChange }
        /> <
        TextField id = "output"
        label = "Output"
        variant = "outlined"
        color = "primary"
        inputProps = {
            {...inputProps, value: data, readOnly: true } }
        onChange = { handleTextInputChange }
        /> <
        /Box> <
        /header> <
        /div>
    );
}

export default App;