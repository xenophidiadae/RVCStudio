using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using RVCStudio.ViewModels;
using RVCStudio.Views;
using RVCStudio.Audio;

namespace RVCStudio;

public partial class App : Application
{
    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var viewModel = new MainViewModel();
            var window = new MainWindow
            {
                DataContext = viewModel,
            };
            window.Closed += (_, _) => viewModel.Dispose();
            desktop.Exit += (_, _) =>
            {
                viewModel.Dispose();
                PortAudioLifetime.Shutdown();
            };
            desktop.MainWindow = window;
        }

        base.OnFrameworkInitializationCompleted();
    }
}
