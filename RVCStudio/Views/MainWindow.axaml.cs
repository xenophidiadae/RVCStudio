using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using RVCStudio.ViewModels;
using System;
using System.Linq;

namespace RVCStudio.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private async void OnDragOver(object? sender, DragEventArgs e)
    {
        var files = await ((Avalonia.Input.IAsyncDataTransfer)e.DataTransfer).TryGetFilesAsync();
        e.DragEffects = files?.Any(file => file.Name.EndsWith(".zip", StringComparison.OrdinalIgnoreCase)) == true
            ? DragDropEffects.Copy : DragDropEffects.None;
    }

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        if (DataContext is not MainViewModel viewModel) return;
        try
        {
            var files = await ((Avalonia.Input.IAsyncDataTransfer)e.DataTransfer).TryGetFilesAsync();
            var file = files?.FirstOrDefault(item => item.Name.EndsWith(".zip", StringComparison.OrdinalIgnoreCase));
            if (file is not null) await viewModel.ImportArchive(file.TryGetLocalPath());
        }
        catch (Exception ex)
        {
            viewModel.StatusMessage = $"Could not process dropped file: {ex.Message}";
        }
    }
}
